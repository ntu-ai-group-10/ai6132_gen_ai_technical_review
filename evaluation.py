import os
import argparse
import json
import yaml
from PIL import Image
from tqdm import tqdm
from src.evaluation_metrics import compute_fid, compute_clip_score

def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    #list down parameter combinations in output directory
    output_dir = cfg['output_dir']
    param_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    
    #iterate over each parameter setting
    for param in param_dirs:
        param_path = os.path.join(output_dir, param)
        print(f"\nEvaluating parameter setting: {param}")
        
        real_images = []
        fake_images = []
        prompts = []
        
        #load images and prompts
        for item in tqdm(os.listdir(param_path)):
            item_path = os.path.join(param_path, item)
            if os.path.isdir(item_path):
                for fname in os.listdir(item_path):
                    if fname.endswith(".png"):
                        img = Image.open(os.path.join(item_path, fname)).convert("RGB")
                        fake_images.append(img)
                        
                        #load corresponding prompt
                        prompt_file = fname.replace(".png", ".txt")
                        with open(os.path.join(item_path, prompt_file), "r", encoding="utf-8") as pf:
                            prompt = pf.read().strip()
                            prompts.append(prompt)

                        #load real image
                        img = Image.open(os.path.join(cfg['input_dir'], f"{item}.jpg")).convert("RGB")
                        real_images.append(img)
                
        # Compute FID
        fid_score = compute_fid(real_images, fake_images)
        
        # Compute CLIP Score
        clip_scores = compute_clip_score(fake_images, prompts)
        avg_clip_score = sum(clip_scores) / len(clip_scores) if clip_scores else 0

        # Path to save evaluation results
        eval_file = os.path.join(output_dir, "evaluation.txt")

        with open(eval_file, "a", encoding="utf-8") as f:
            f.write(f"Parameter setting: {param}\n")
            f.write(f"FID Score: {fid_score}\n")
            f.write(f"Average CLIP Score: {avg_clip_score}\n")
            f.write("="*100 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    main(args)