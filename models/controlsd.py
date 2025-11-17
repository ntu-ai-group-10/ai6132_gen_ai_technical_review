import torch
import numpy as np
from PIL import Image
from transformers import pipeline
from diffusers.utils import load_image
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)


class StableDiffusionControlNetGenerator:
    """
    A class that encapsulates the Stable Diffusion v1.5 + ControlNet (Depth) model.
    It automatically performs depth map estimation and preprocessing before generation.
    """
    def __init__(self, 
                 controlnet_checkpoint: str = "lllyasviel/control_v11f1p_sd15_depth", 
                 base_model: str = "runwayml/stable-diffusion-v1-5"):
        
        # 1. Initialize the depth estimation pipeline (runs on CPU by default)
        self.depth_estimator = pipeline('depth-estimation')

        # 2. Initialize the ControlNet model
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_checkpoint, 
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        
        # 3. Initialize the Stable Diffusion + ControlNet generation pipeline
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model, 
            controlnet=self.controlnet, 
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        
        # 4. Configure the scheduler and optimization
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        # Enable model offloading to CPU to save VRAM (at the cost of some latency)
        #self.pipe.enable_model_cpu_offload()

    def _preprocess_depth_map(self, input_image: Image.Image) -> Image.Image:
        """
        Estimates the depth map from the input image and preprocesses it 
        to be suitable as a ControlNet input (converting it to a 3-channel image).
        
        Args:
            input_image: The original image (PIL Image) for depth estimation.
            
        Returns:
            The processed control image (depth map) ready for ControlNet.
        """
        
        # Estimate the depth map (returns a dictionary containing the depth as a PIL Image)
        depth_map = self.depth_estimator(input_image)['depth']
        
        # Convert the PIL Image to a NumPy array for manipulation
        image_array = np.array(depth_map)
        
        # Expand the grayscale map (H, W) to (H, W, 1) and concatenate it to (H, W, 3) 
        # as ControlNet expects a 3-channel input image.
        image_array = image_array[:, :, None]
        control_image_array = np.concatenate([image_array, image_array, image_array], axis=2)
        
        # Convert back to PIL Image format
        control_image = Image.fromarray(control_image_array)
        
        return control_image

    def generate(self, prompt: str, input_image: Image.Image, **kwargs) -> Image.Image:
        """
        Generates an image based on the text prompt, guided by the depth map 
        derived from the input image.
        
        Args:
            prompt: The text prompt for generation.
            input_image: The input image (PIL Image) used to calculate the depth control map.
            **kwargs: Optional generation parameters (e.g., num_inference_steps, seed, guidance_scale).
            
        Returns:
            The generated image (PIL Image).
        """
        # 1. Preprocessing: Generate the depth control image for ControlNet
        control_image = self._preprocess_depth_map(input_image)
        
        # 2. Set the torch generator for reproducible results (seed)
        seed = kwargs.get("seed", 0)
        generator = torch.manual_seed(seed)
        
        # 3. Run the pipeline to generate the final image
        generated_image = self.pipe(
            prompt, 
            image=control_image,  # The preprocessed depth map
            num_inference_steps=kwargs.get("num_inference_steps", 30),
            guidance_scale=kwargs.get("guidance_scale", 7.5),
            max_sequence_length=kwargs.get("max_sequence_length", 512),
            generator=generator,
        ).images[0]

        return generated_image

# --- Example Usage ---

# 1. Load the input image from the specified URL
if __name__ == "__main__":
    input_url = "https://huggingface.co/lllyasviel/control_v11p_sd15_depth/resolve/main/images/input.png"
    input_image = load_image(input_url)
    
    # 2. Initialize the generator (This loads the heavy models)
    generator_instance = StableDiffusionControlNetGenerator()
    
    # 3. Define the prompt
    prompt_text = "Stormtrooper's lecture in beautiful lecture hall"
    
    # 4. Generate the image
    output_image = generator_instance.generate(
        prompt=prompt_text, 
        input_image=input_image, 
        num_inference_steps=30,
        seed=0
    )

    import os 
    # Ensure the 'images' directory exists for saving example output
    os.makedirs("images", exist_ok=True) 
    # 5. Save the result
    output_image.save('image/image_out.png') 
    print("Image saved to images/image_out.png")