import os
from datasets import load_dataset
import aiohttp
from PIL import Image
import json

if __name__ == "__main__":
    # #load dataset
    dataset= load_dataset("Adyakanta/test_flickr30k",split="test")
    #prepare output folder
    output_dir = "data/flickr"
    os.makedirs(output_dir, exist_ok=True)
    
    #save images and captions
    metadata = []

    for i, item in enumerate(dataset):
        id = item['img_id']+'.jpg'
        image: Image.Image = item["image"]
        img_path = os.path.join(output_dir, id)
        image.save(img_path, format="JPEG")

        caption = item["caption"]
        metadata.append({
            "id": id,
            "caption": caption
        })

        if i%500 == 0:
            print(f"Saved {i} samples...")
    
    # Save metadata JSON
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print("Dataset flickr30k preparation completed.")



    #load dataset
    dataset = load_dataset(
        "lmms-lab/COCO-Caption2017",
        split="val",
    )

    #prepare output folder
    output_dir = "data/coco"
    os.makedirs(output_dir, exist_ok=True)
    
    #save images and captions
    metadata = []
    for i, item in enumerate(dataset):
        id = item['file_name']
        image: Image.Image = item["image"]
        img_path = os.path.join(output_dir, id)
        image.save(img_path, format="JPEG")

        caption = item["answer"]
        metadata.append({
            "id": id,
            "caption": caption
        })

        if i%500 == 0:
            print(f"Saved {i} samples...")
    
    # Save metadata JSON
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print("Dataset coco preparation completed.")

