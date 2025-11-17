import os
import argparse
from models.stable_diffusion import StableDiffusionGenerator
from models.controlsd import StableDiffusionControlNetGenerator
import json
import random
import yaml
import itertools

import os
from PIL import Image
from typing import Optional

def load_image_by_id(image_id: str, image_folder: str) -> Optional[Image.Image]:
    full_path = os.path.join(image_folder, image_id)
    try:
        image = Image.open(full_path)
        return image.convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at path: {full_path}")
        return None
    except Exception as e:
        print(f"Error loading image {image_id}: {e}")
        return None
        
MODEL_MAP = {
    "stable_diffusion": StableDiffusionGenerator,
    "controlsd": StableDiffusionControlNetGenerator
}

def expand_grid(param_dict):
    """Turns {a:[1,2], b:[3,4]} -> [{a:1,b:3},{a:1,b:4},{a:2,b:3},{a:2,b:4}]"""
    keys = param_dict.keys()
    vals = param_dict.values()
    combinations = itertools.product(*vals)
    return [dict(zip(keys, combo)) for combo in combinations]

def main(args):
    
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Read prompts JSON
    with open(cfg['prompts'], 'r') as f:
        prompts = json.load(f)

    #Create output directory
    os.makedirs(cfg['output_dir'], exist_ok=True)

    #Load model class
    generator = MODEL_MAP[cfg['model']]()

    #Create parameter grid
    param_grid = expand_grid(cfg["params"])
    print(f"Total runs: {len(param_grid)}")

    # Apply sample count
    if cfg['sample_count'] is not None:
        prompts = prompts[:cfg['sample_count']]
        print(f"Using only the first {cfg['sample_count']} samples.")

    is_controlnet_mode = bool(cfg.get('controlnet_enabled', 0))

    IMAGE_FOLDER = cfg.get('input_dir',None)
    
    for i, params in enumerate(param_grid):
        print(f"\n=== Run {i+1}/{len(param_grid)} with params: {params} ===")
        params_id = params.__str__().replace(" ", "").replace(":", "=").replace(",", "_").replace("{", "").replace("}", "")
        params_output_dir = os.path.join(cfg['output_dir'], params_id)

        
        #Generate images for each prompt
        for item in prompts:
            img_id = item["id"]
            captions = item["caption"]
        
            if cfg['caption_strategy'] == "first":
                selected_captions = [captions[0]]
            elif cfg['caption_strategy'] == "random":
                selected_captions = [random.choice(captions)]
            else:  # "all"
                selected_captions = captions
        
            #  control net
            input_image = None
            if is_controlnet_mode:
                input_image = load_image_by_id(img_id, IMAGE_FOLDER)
                
                # if not exist
                if input_image is None:
                    print(f"Skipping {img_id}: Could not load image for ControlNet.")
                    continue

                
                
            # Create a folder for this image id
            img_output_dir = os.path.join(params_output_dir, img_id.replace(".jpg", ""))
            os.makedirs(img_output_dir, exist_ok=True)

            for idx, prompt in enumerate(selected_captions):
                print(f"Generating image for {img_id} | prompt {idx}: {prompt}")
        
                # control net: input_image
                if is_controlnet_mode:
                    img = generator.generate(prompt, input_image=input_image,**params)
                else:
                    img = generator.generate(prompt,**params)
        
                out_path = os.path.join(img_output_dir, f"{idx}.png")
                img.save(out_path)
        

                # save the prompt as a text file
                prompt_path = os.path.join(img_output_dir, f"{idx}.txt")
                with open(prompt_path, "w+", encoding="utf-8") as f:
                    f.write(prompt)

                print(f"Saved → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    main(args)