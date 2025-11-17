import os
import torch
from PIL import Image
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoModelForImageClassification, AutoImageProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------
# 1. FID CALCULATION
# ---------------------------------------------------
def compute_fid(real_imgs, fake_imgs):
    fid = FrechetInceptionDistance(normalize=True).to(device)

    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    # Add real images
    for img in real_imgs:
        fid.update(preprocess(img).unsqueeze(0).to(device), real=True)

    # Add generated images
    for img in fake_imgs:
        fid.update(preprocess(img).unsqueeze(0).to(device), real=False)

    return float(fid.compute())


# ---------------------------------------------------
# 2. CLIP-SCORE (text-image similarity)
# ---------------------------------------------------
def compute_clip_score(images, prompts):
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    scores = []
    for img, prompt in zip(images, prompts):
        inputs = processor(text=[prompt], images=img, return_tensors="pt", padding=True).to(device)
        outputs = clip_model(**inputs)
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds  = outputs.text_embeds  / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
        score = (image_embeds @ text_embeds.T).item()
        scores.append(score)

    return scores