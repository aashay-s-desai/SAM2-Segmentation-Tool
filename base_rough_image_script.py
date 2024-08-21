import torch
import os
import time
import json
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.build_sam import build_sam2_video_predictor
from PIL import Image
import numpy as np

# Paths to the checkpoint and model configuration
checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

# Build the SAM2 model for video prediction
video_predictor = build_sam2_video_predictor(model_cfg, checkpoint)


# Define the video path
video_path = "/home/earthsense/Documents/TEST_IMAGE_FOLDER"


output_folder = "/home/earthsense/segmented_video_frames"
# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize the inference state
inference_state = video_predictor.init_state(video_path)

# Collect click coordinates and labels (this can be adapted to handle video clicks if necessary)
click_coords = []
point_labels = []

def on_click(event):
    x = int(event.xdata)
    y = int(event.ydata)
    click_coords.append((x, y))
    point_labels.append(1)
    print(f"Point: ({x}, {y})")

# Load and display the first frame to collect user clicks
fig, ax = plt.subplots()
first_image = inference_state['images'][0].cpu().numpy().transpose(1, 2, 0) #have to use .cpu() cuz some python functions can't access GPU, .numpy() so i can do stuff w matplotlib n stuff, .transpose(1, 2, 0), cuz pytorch format is different from matplotlib format for imshow
first_image = np.clip(first_image, 0, 1)  # Ensure image data is within the correct range
ax.imshow(first_image)
cid = fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()

# Convert click coordinates and labels to the required format
point_coords = torch.tensor(click_coords, dtype=torch.float32)
point_labels = torch.tensor(point_labels, dtype=torch.int32)

# Add initial points or box to the first frame (frame index 0)
frame_idx, obj_ids, video_res_masks = video_predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=0,  # Start from the first frame
    obj_id=0,  # Object ID (if you are tracking multiple objects, you can manage different IDs)
    points=point_coords.unsqueeze(0),  # Add batch dimension
    labels=point_labels.unsqueeze(0),  # Add batch dimension
)

# Measure FPS for processing the video
start_time = time.time()

# Propagate the masks across the video
video_masks = []
for frame_idx, obj_ids, video_res_masks in video_predictor.propagate_in_video(
        inference_state=inference_state, start_frame_idx=0):
    video_masks.append(video_res_masks.cpu().numpy())



# Save segmentation data as a JSON file
segmentation_data = {
    "video_masks": [mask.tolist() for mask in video_masks],
    "obj_ids": obj_ids,
    "frame_indices": list(range(len(video_masks)))
}

output_json_path = "/home/earthsense/segment-anything-2/video_segmentation_data.json"
with open(output_json_path, 'w') as f:
    json.dump(segmentation_data, f, indent=4)

# Calculate FPS
end_time = time.time()
processing_time = end_time - start_time
fps = len(video_masks) / processing_time
print(f"Processing Time: {processing_time}")
print(f"FPS: {fps:.2f}")

# Convert segmented image files to video and display video
plt.figure(figsize=(10, 10))
last_image = inference_state['images'][-1].cpu().numpy().transpose(1, 2, 0)
last_image = np.clip(last_image, 0, 1)  # Ensure image data is within the correct range
plt.imshow(first_image)
for mask in video_res_masks:
    plt.imshow(mask.cpu().numpy().transpose(1, 2, 0), alpha=0.5) #WHY NO NP.CLIP HERE: masks are likely binary or probability maps, meaning their values should already be in a range that matplotlib can handle ([0, 1] for floats or [0, 255] for integers).
plt.axis('off')
plt.savefig("segmented_video_frame.png", bbox_inches='tight', pad_inches=0)
plt.show()

print(f"Segmented video data saved to {output_json_path}")


'''
# Convert segmented image files to video and display video
plt.figure(figsize=(10, 10))
plt.imshow(inference_state['images'][-1].cpu().numpy().transpose(1, 2, 0))
for mask in video_res_masks:
    plt.imshow(mask.cpu().numpy().transpose(1, 2, 0), alpha=0.5)

plt.axis('off')
plt.savefig("segmented_video_frame.png", bbox_inches='tight', pad_inches=0)
plt.show()

print(f"Segmented video data saved to {output_json_path}")
'''

