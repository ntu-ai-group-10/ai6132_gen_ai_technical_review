import os
import argparse
from models.stable_diffusion import StableDiffusionGenerator
import json
import random

MODEL_MAP = {
    "stable_diffusion": StableDiffusionGenerator,
}

def main(args):
    # Read prompts JSON
    with open(args.prompts, 'r') as f:
        prompts = json.load(f)

    #Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    #Load model class
    generator = MODEL_MAP[args.model]()

    #Generate images for each prompt
    for item in prompts:
        img_id = item["id"]
        captions = item["caption"]

        # Select captions based on strategy
        if args.caption_strategy == "first":
            selected_captions = [captions[0]]
        elif args.caption_strategy == "random":
            selected_captions = [random.choice(captions)]
        else:  # "all"
            selected_captions = captions
        
        # Create a folder for this image id
        img_output_dir = os.path.join(args.output_dir, img_id.replace(".jpg", ""))
        os.makedirs(img_output_dir, exist_ok=True)

        for idx, prompt in enumerate(selected_captions):
            print(f"Generating image for {img_id} | prompt {idx}: {prompt}")

            img = generator.generate(prompt)

            out_path = os.path.join(img_output_dir, f"{idx}.png")
            img.save(out_path)

            print(f"Saved → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--caption_strategy", type=str, default="all", choices=["first", "random", "all"], help="Choose how to pick captions")
    args = parser.parse_args()

    main(args)