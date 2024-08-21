import os
import cv2
import torch
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import supervision as sv  # Supervision library for easier handling
from pathlib import Path
from sam2.build_sam import build_sam2_video_predictor
from jupyter_bbox_widget import BBoxWidget

# Define paths and configurations
HOME = os.getcwd()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = f"{HOME}/checkpoints/sam2_hiera_large.pt"
CONFIG = "sam2_hiera_l.yaml"
input_video_path = "/home/earthsense/Documents/TEST_IMAGE_FOLDER"
segmented_video_frames = Path(HOME) / "segmented_video_frames"
segmented_video_frames.mkdir(parents=True, exist_ok=True)

# Build the SAM2 model for video prediction
video_predictor = build_sam2_video_predictor(CONFIG, CHECKPOINT)

# Initialize the inference state
inference_state = video_predictor.init_state(video_path=input_video_path)

# Collect click coordinates and labels (can be adapted to handle video clicks)
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
first_image = inference_state['images'][0].cpu().numpy().transpose(1, 2, 0)
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

# Set the scale factor and video processing settings
SCALE_FACTOR = 0.5
START_IDX = 0
END_IDX = len(inference_state['images']) - 1  # Process all frames

# Measure FPS for processing the video
start_time = time.time()

# Propagate the masks across the video
video_masks = []
for frame_idx, obj_ids, video_res_masks in video_predictor.propagate_in_video(
        inference_state=inference_state, start_frame_idx=0):

    # Resize masks to match the original image dimensions
    original_frame = inference_state['images'][frame_idx].cpu().numpy().transpose(1, 2, 0)
    original_frame = np.clip(original_frame, 0, 1)
    
    resized_masks = [cv2.resize(mask.cpu().numpy().squeeze(), 
                                (original_frame.shape[1], original_frame.shape[0]))
                     for mask in video_res_masks]
    
    video_masks.append(resized_masks)

    # Save image in image folder so I can convert to video later
    plt.figure(figsize=(10, 10))
    plt.imshow(original_frame)

    # Overlay resized masks
    for mask in resized_masks:
        plt.imshow(mask, cmap='gray', alpha=0.5)
    
    output_image_path = segmented_video_frames / f"{frame_idx:06d}_segmented.jpg"
    plt.axis('off')
    plt.savefig(output_image_path.as_posix(), bbox_inches='tight', pad_inches=0)
    plt.close()

print(f"Segmented video frames saved to {segmented_video_frames}")

# Save segmentation data as a JSON file
segmentation_data = {
    "video_masks": [mask.tolist() for mask in video_masks],
    "obj_ids": obj_ids,
    "frame_indices": list(range(len(video_masks)))
}

segmentation_json_path = Path(HOME) / "video_segmentation_data.json"
with open(segmentation_json_path, 'w') as f:
    json.dump(segmentation_data, f, indent=4)

# Calculate FPS
end_time = time.time()
processing_time = end_time - start_time
fps = len(video_masks) / processing_time
print(f"Processing Time: {processing_time}")
print(f"FPS: {fps:.2f}")

# Convert segmented image files to video
output_video_path = Path(HOME) / "output_video.mp4"
first_seg_frame = cv2.imread(str(segmented_video_frames / f"000000_segmented.jpg"))
height, width, _ = first_seg_frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(str(output_video_path), fourcc, fps=30, frameSize=(width, height))

# Write frames to the video
for image_path in sorted(segmented_video_frames.glob("*.jpg")):
    frame = cv2.imread(str(image_path))
    video.write(frame)

video.release()
print(f"Output video saved to {output_video_path}")

# Display video (optional)
os.system(f'xdg-open {output_video_path}')
