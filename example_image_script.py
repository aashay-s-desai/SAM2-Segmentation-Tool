import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Load the image
image_path = "/home/earthsense/Documents/collection-130624_040630_zed_camera_vis/000003.jpg"
image = Image.open(image_path).convert("RGB")

# Convert image to a format compatible with SAM2 (numpy array)
image_np = np.array(image)


# Paths to the checkpoint and model configuration
checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

# Create the image predictor
sam2_model = build_sam2(
    config_file=model_cfg,
    ckpt_path=checkpoint,
    device="cuda",
    mode="eval"
)
predictor = SAM2ImagePredictor(sam2_model)



#Get point coordinates
click_coords = []
point_labels = []
def on_click(event):
    x = int(event.xdata)
    y = int(event.ydata)
    click_coords.append((x, y))
    point_labels.append(1)
    print(f"Point: ({x}, {y})")

# Display the image
fig, ax = plt.subplots()
ax.imshow(image_np)
# Connect event handler
cid = fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()



# Set point coorinates and labels to guide mask prediction
point_coords = np.array(click_coords)
point_labels = np.array(point_labels)

# Set bounding box (optional)
#box = np.array([50, 50, 200, 200])

# Inference with SAM2
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image_np)

    # Define prompts for the trunk areas using your captured points
    '''
    prompts = {
        "point_coords": [[434, 172], [133, 153], [242, 149], [267, 157], [373, 164], [529, 172], [578, 171]],
        "point_labels": [1, 1, 1, 1, 1, 1, 1],  # 1 for foreground (tree trunks)
	"box": np.array([50, 50, 150, 150])
    }
    
    print(prompts)
    '''

    # Predict masks
    masks, ious, low_res_masks = predictor.predict(
        point_coords=point_coords,
	point_labels=point_labels,
    )
    
'''   
mask_generator = SAM2AutomaticMaskGenerator(model=sam2_model)
auto_masks = mask_generator.generate(image.np)
'''


print("Predicted Masks (Manual):", masks)
print("IoU Predictions (Manual):", ious)
print("Low-Resolution Masks (Manual):", low_res_masks)
#print("\nGenerated Masks (Automatic):", auto_masks)


#Visualize the result
plt.figure(figsize=(10,10))
plt.imshow(image_np)


for mask in masks:
    plt.imshow(mask, alpha=0.5)
    
plt.axis('off')
plt.show()
