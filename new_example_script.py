import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the image (replace with your image path)
image_path = '/home/earthsense/Documents/person_with_dog.jpg'
image = Image.open(image_path).convert("RGB")

# Set up the predictor with a specific checkpoint
checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Use the predictor to set the image
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)

    # Generate the masks
    masks, _, _ = predictor.predict()

# Assuming masks[0] is the mask you want to save
# Convert the mask (numpy array) to a PIL Image and save it
mask_image = Image.fromarray((masks[0] * 255).astype(np.uint8))  # Ensure it's in uint8 format
mask_image.save('/home/earthsense/Documents/segmented_dog_with_man.png')

# Optionally, display the mask
plt.imshow(mask_image, cmap='gray')
plt.show()


