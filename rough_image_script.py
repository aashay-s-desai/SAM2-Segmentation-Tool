import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import time
import json
import os

# Paths to the checkpoint and model configuration
checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"


'''
# Create the image predictor
sam2_model = build_sam2(
    config_file=model_cfg,
    ckpt_path=checkpoint,
    device="cuda",
    mode="eval"
)
'''

predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))




# Load the image
#image_path = "/home/earthsense/Documents/person_with_dog.jpg"
image_path = "/home/earthsense/Documents/collection-130624_040630_zed_camera_vis/000003.jpg"
image = Image.open(image_path).convert("RGB")

# Convert image to a format compatible with SAM2 (numpy array)
image_np = np.array(image)


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



# Measure FPS for processing the image
start_time = time.time()


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


# Save segmentation data as a JSON file
segmentation_data = {
    "masks": [mask.tolist() for mask in masks],
    "ious": ious.tolist(),
    "low_res_masks": [low_res_mask.tolist() for low_res_mask in low_res_masks]
}

with open("/home/earthsense/segment-anything-2/image_segmentation_data.json", 'w') as f:
    json.dump(segmentation_data, f, indent=4)




# Calculate FPS
end_time = time.time()
processing_time = end_time - start_time
fps = 1 / processing_time
print(f"Processing Time: {processing_time}")
print(f"FPS: {fps}")
print(f"FPS: {fps:.2f}")


print("Predicted Masks (Manual):", masks)
print("IoU Predictions (Manual):", ious)
print("Low-Resolution Masks (Manual):", low_res_masks)
#print("\nGenerated Masks (Automatic):", auto_masks)



# Visualize the result
plt.figure(figsize=(10,10))
plt.imshow(image_np)

# Drape masks over original image
for mask in masks:
    plt.imshow(mask, alpha=0.5) #cmap colors if ur picky lol (add arg " cmap='winter' " for example): https://matplotlib.org/stable/users/explain/colors/colormaps.html
    
 
plt.axis('off') #turn off axis' and other graph stuff


# Save image
plt.savefig("segmented_image.jpg", bbox_inches='tight', pad_inches=0)

# Display image
plt.show()


'''
# Automatic file/folder name system
# Check if the directory exists, and if not, create it
if not os.path.exists("/home/earthsense/segment-anything-2/segmented_images_folder):
    os.makedirs("segmented_images_folder")  # Create the directory
os.path.join...


#LINUX command of folder of images to mp4: ffmpeg -i /home/earthsense/Documents/collection-130624_040630_zed_camera_vis/%06d.jpg /home/earthsense/Documents/VIDEO_collection-130624_040630_zed_camera_vis.mp4^C

'''
