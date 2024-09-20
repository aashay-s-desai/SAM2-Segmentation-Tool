import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2_video_predictor

import cv2

#fps
import time

import json

# Paths to the checkpoint and model configuration
checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

# Build the SAM2 model for video prediction
video_predictor = build_sam2_video_predictor(model_cfg, checkpoint)




def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))













# Define the video path
input_video_path = "/home/earthsense/Documents/collection-130624_040630_zed_camera_vis_short"


# Define the output video frames folder path
segmented_video_frames = "/home/earthsense/segment-anything-2/segmented_video_frames"
# Create the output folder if it doesn't exist
os.makedirs(segmented_video_frames, exist_ok=True)


# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(input_video_path)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))


inference_state = video_predictor.init_state(video_path=input_video_path)





# Collect click coordinates and labels (this can be adapted to handle video clicks if necessary)
point_coords = []
point_labels = []
prompts = {}
ann_obj_id = 1
ann_frame_idx = 0


#initializing these and making them global variables so i can print out an image of everything totally segmented at the end
final_out_frame_idx = 0
final_out_obj_ids = []
final_out_video_res_masks = torch.empty(0)


def on_click(event):
    global ann_obj_id

    global point_coords
    global point_labels
    global prompts


    global final_out_frame_idx
    global final_out_obj_ids
    global final_out_video_res_masks


    x = int(event.xdata)
    y = int(event.ydata)

    # Different click events
    if (event.button == 1): # Left Click
        label = 1 # Positive Click
    elif (event.button == 3): # Right Click
        label = 0 # Negative Click
    elif (event.button == 2): # NEW OBJECT (new obj_id)
        ann_obj_id+=1
        print(f"Starting a new object with obj_id: {ann_obj_id}")
        return  # Exit so that no coordinates are recorded for the middle click
        #Every time there's a new obj_id
    else:
        return # ignore other buttons
    
    point_coords.append((x, y))
    point_labels.append(label)
    click_type = "Positive" if label == 1 else "Negative"
    print(f"{click_type} Click")
    print(f"Click: ({x}, {y}) with label: {label} for obj_id: {ann_obj_id}") 


    # Store the coordinates and labels for the current object ID
    if ann_obj_id not in prompts:
        prompts[ann_obj_id] = ([], [])
    prompts[ann_obj_id][0].append((x,y))
    prompts[ann_obj_id][1].append(label)

    # Convert click coordinates and labels to the required format
    point_coords_np = np.array(prompts[ann_obj_id][0], dtype=np.float32)
    point_labels_np = np.array(prompts[ann_obj_id][1], dtype=np.int32)


    # Add initial points or box to the first frame (frame index 0)

    #SAVE out_frame_idx, out_obj_ids, and out_video_res_masks so i can print out that last culminating image at the end
    out_frame_idx, out_obj_ids, out_video_res_masks = video_predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,  # Start from the first frame
        obj_id=ann_obj_id,  # Object ID (if you are tracking multiple objects, you can manage different IDs)
        points=point_coords_np,
        labels=point_labels_np
    )

    final_out_frame_idx, final_out_obj_ids, final_out_video_res_masks = out_frame_idx, out_obj_ids, out_video_res_masks


    # Show the results on the current (interacted) frame
    plt.gca().cla()  # Clear the current axes 
    plt.title(f"Frame {out_frame_idx} with Segmentation")
    plt.imshow(Image.open(os.path.join(input_video_path, frame_names[out_frame_idx])))

    show_points(point_coords_np, point_labels_np, plt.gca())


    # Ensure the mask is on the CPU before converting to NumPy
    for i, out_obj_id in enumerate(out_obj_ids):
        show_mask((out_video_res_masks[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
    plt.show()


# Load and display the first frame to collect user clicks
fig, ax = plt.subplots(figsize=(9, 6))
first_image = Image.open(os.path.join(input_video_path, frame_names[0]))

ax.imshow(first_image)
cid = fig.canvas.mpl_connect('button_press_event', on_click)
plt.title('Click on the image to begin selecting points (clicks: right -> pos, left -> neg, middle -> new object)')
plt.show()


# Show the FINAL results on a new frame
plt.figure(figsize=(9, 6))
plt.title(f"Frame {final_out_frame_idx} with Segmentation")
current_image = Image.open(os.path.join(input_video_path, frame_names[final_out_frame_idx]))
plt.imshow(current_image)

#since the global variables are non-converted
point_coords_np = np.array(prompts[ann_obj_id][0], dtype=np.float32) #the reason you do prompts[ann_obj_id][0] (and same for lables) instead of just point_coords is because for another obj_id, you might have overlapping point coordinates, but you still want to process them as separate objects to segment
point_labels_np = np.array(prompts[ann_obj_id][1], dtype=np.int32)

show_points(point_coords_np, point_labels_np, plt.gca())

# Ensure the mask is on the CPU before converting to NumPy
for i, out_obj_id in enumerate(final_out_obj_ids):
    show_mask((final_out_video_res_masks[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)  #THE REASON WHY EACH OBJECT ID HAS A UNIQUE COLOR VALUE IS DUE TO THE SHOW_MASK FUNCTION, WHICH ASSIGNS A UNIQUE COLOR PER EACH UNIQUE OBJECT ID
plt.show()






# Measure FPS for processing the video
start_time = time.time()

#Run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
segmentation_data = {} # Initialize segmentation data storage
for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

    # Save segmentation data for this frame
    segmentation_data[out_frame_idx] = []
    for i, out_obj_id in enumerate(out_obj_ids):
        mask = (out_mask_logits[i] > 0.0).cpu().numpy().tolist()
        segmentation_data[out_frame_idx].append({
            'obj_id': int(out_obj_id),
            'mask': mask
        })

# render the segmentation results every few frames
vis_frame_stride = 1
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(input_video_path, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

    output_image_path = os.path.join(segmented_video_frames, f"{out_frame_idx:06d}_segmented.jpg")
    plt.axis('off')
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
# plt.show()
print(f"Segmented video frames saved to {segmented_video_frames}")


# Calculate FPS
end_time = time.time()
processing_time = end_time - start_time
fps = len(frame_names) / processing_time
print(f"Processing Time: {processing_time}")
print(f"FPS: {fps:.2f}")

# Save the final segmentation data to a JSON file
output_json_path = "/home/earthsense/segment-anything-2/segmentation_info.json"
with open(output_json_path, 'w') as json_file:
    json.dump(segmentation_data, json_file, indent=4)

print(f"Segmentation data saved to {output_json_path}")






#DISPLAY VIDEO
# Convert segmented image files to video and display video
#save list of segmented image paths from segmented_video_frames folder
output_path_list = sorted([image for image in os.listdir(segmented_video_frames)]) #don't need to sort should already be sorted


first_seg_frame_path = os.path.join(segmented_video_frames, output_path_list[0])
first_seg_frame_loaded = cv2.imread(first_seg_frame_path) 
# Get the width and height of the first segmented frame
height, width, layers = first_seg_frame_loaded.shape
    

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4
output_video_path = "/home/earthsense/segment-anything-2/output_video.mp4"
video = cv2.VideoWriter(output_video_path, fourcc, fps=30, frameSize=(width, height))
    
for image_path in output_path_list:
    full_image_path = os.path.join(segmented_video_frames, image_path)
    frame = cv2.imread(full_image_path)
    video.write(frame)
    
video.release()
cv2.destroyAllWindows()
print(f"Output video saved to {output_video_path}")

#display video
os.system(f'xdg-open {output_video_path}')