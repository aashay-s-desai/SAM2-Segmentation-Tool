import os
from pathlib import Path
import torch
from sam2.build_sam import build_sam2
#from sam2.utils import predict_video
from PIL import Image

# Set up the SAM2 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths to the checkpoint and model configuration
checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

# Build the SAM2 model for video prediction
#video_predictor = build_sam2_video_predictor(model_cfg, checkpoint)

model = build_sam2(model_cfg, checkpoint, device=device)

# Define the directory containing the images
image_dir = Path('/home/earthsense/Documents/collection-130624_040630_zed_camera_vis')

# Get all image paths
image_paths = sorted(image_dir.glob("*.jpg"))  # Change the extension if your images are not .jpg

# Run SAM2 prediction on the images
output_dir = Path('/home/earthsense/Documents/collection-130624_040630_zed_camera_vis/output')
output_dir.mkdir(exist_ok=True)

for idx, image_path in enumerate(image_paths):
    print(f"Processing image {idx+1}/{len(image_paths)}: {image_path}")
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Run prediction
    segmentation_mask = model.predict(image)

    # Save the result
    output_path = output_dir / f"{image_path.stem}_mask.png"
    segmentation_mask.save(output_path)

print("SAM2 video prediction completed.")
