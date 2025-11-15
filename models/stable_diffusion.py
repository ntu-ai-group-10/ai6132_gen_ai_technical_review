from diffusers import StableDiffusionPipeline
import torch

class StableDiffusionGenerator:
    def __init__(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

    def generate(self, prompt, **kwargs):
        return self.pipe(
            prompt,
            num_inference_steps=12,
            guidance_scale=3.0,
            max_sequence_length=512,
        ).images[0]