import torch

#accessing/creating folders/files
import os

#fps
import time

#saving segmentation data and other stuff idfr
import json
import matplotlib.pyplot as plt
import numpy as np

#sam2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.build_sam import build_sam2_video_predictor

#for handling image/image processing
from PIL import Image

#for creating video
import cv2
#also os for displaying video


# Paths to the checkpoint and model configuration
checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

# Build the SAM2 model for video prediction
video_predictor = build_sam2_video_predictor(model_cfg, checkpoint)


# Define the video path
input_video_path = "/home/earthsense/Documents/TEST_IMAGE_FOLDER"

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(input_video_path)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))







segmented_video_frames = "/home/earthsense/segment-anything-2/segmented_video_frames"
# Create the output folder if it doesn't exist
os.makedirs(segmented_video_frames, exist_ok=True)

# Initialize the inference state
inference_state = video_predictor.init_state(input_video_path)


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
fig, ax = plt.subplots(figsize=(6.4, 3.6))
first_image = Image.open(os.path.join(input_video_path, frame_names[0]))
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


    # image_with_mask = inference_state['images'][frame_idx].cpu().numpy().transpose(1, 2, 0)
    # image_with_mask = np.clip(image_with_mask, 0, 1)

    # for mask in video_res_masks:
    #     image_with_mask.append
    #     print("Mask Shape:", mask.cpu().numpy().transpose(1, 2, 0).shape)

    # cv2.imwrite(os.path.join(segmented_video_frames, f"{frame_idx:06d}_segmented.jpg"),image_with_mask)

    # Save image in image folder so i can convert to vid later


    # Create and save image
    plt.figure(figsize=(6.4, 3.6))
    image_before_mask = Image.open(os.path.join(input_video_path, frame_names[frame_idx]))
    plt.imshow(image_before_mask)
    
    
    
    

    #print(image_before_mask)
    print("Original Image Shape:", image_before_mask.shape)

    
    for mask in video_res_masks:
        # plt.imshow(mask.cpu().numpy().transpose(1, 2, 0), alpha=0.5) #WHY NO NP.CLIP HERE: masks are likely binary or probability maps, meaning their values should already be in a range that matplotlib can handle ([0, 1] for floats or [0, 255] for integers).
        # print("Mask Shape:", mask.cpu().numpy().transpose(1, 2, 0).shape)

        # resize the mask
        resized_mask = cv2.resize(mask.cpu().numpy().transpose(1, 2, 0), (first_image.shape[1], first_image.shape[0]))  
        
        # overlay the resized mask
        plt.imshow(resized_mask, alpha=0.5)  # Overlay the resized mask with transparency
        
        print("Resized Mask Shape:", resized_mask.shape)


    output_image_path = os.path.join(segmented_video_frames, f"{frame_idx:06d}_segmented.jpg")
    plt.axis('off')
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()



print(f"Segmented video frames saved to {segmented_video_frames}")


# Save segmentation data as a JSON file
segmentation_data = {
    "video_masks": [mask.tolist() for mask in video_masks],
    "obj_ids": obj_ids,
    "frame_indices": list(range(len(video_masks)))
}

segmentation_json_path = "/home/earthsense/segment-anything-2/video_segmentation_data.json"
with open(segmentation_json_path, 'w') as f:
    json.dump(segmentation_data, f, indent=4)

# Calculate FPS
end_time = time.time()
processing_time = end_time - start_time
fps = len(video_masks) / processing_time
print(f"Processing Time: {processing_time}")
print(f"FPS: {fps:.2f}")

# Convert segmented image files to video and display video
#save list of segmented image paths from segmented_video_frames folder
output_path_list = [image for image in os.listdir(segmented_video_frames)] #don't need to sort should already be sorted


first_seg_frame_path = os.path.join(segmented_video_frames, output_path_list[0])
first_seg_frame_loaded = cv2.imread(first_seg_frame_path) 
# Get the width and height of the first segmented frame
height, width, layers = first_seg_frame_loaded.shape
    

# Define the codec and create VideoWriter object (fourcc means 4-character-code of format of codec; codec determines how the video stream is compressed or decompressed)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4
output_video_path = "/home/earthsense/segment-anything-2/output_video.mp4"
video = cv2.VideoWriter(output_video_path, fourcc, fps=30, frameSize=(width, height))
    
for image_path in output_path_list:
    full_image_path = os.path.join(segmented_video_frames, image_path)
    frame = cv2.imread(full_image_path)
    video.write(frame)
    
video.release()
cv2.destroyAllWindows()
print(f"Segmented video data saved to {segmentation_json_path}")
print(f"Output video saved to {output_video_path}")

#display video
os.system(f'xdg-open {output_video_path}')



'''
MAIN PROBLEMS RIGHT NOW:
 - when i run base_rough... it propogates through all 10 images together and displays that _C error message only once, and rough_video_script.py it generates that _C error between each image propogation
 - when selecting points in first image, its weirdly shaped and colored
 - in segmented images, its zoomed, and doesn't seem to be even working idk


FUTURE:
 - try with drawing bounding boxes instead of points
 - try processing each image in video instead of first frame then auto-process like how SAM2 typically does, and see how intensive it is on the computer (GPU, CPU, FPS, etc.)
 - make streamlit app for whicever of the three i wanna move on with



'''