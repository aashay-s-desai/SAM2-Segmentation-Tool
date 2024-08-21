import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image
import numpy as np

# Load the image
image_path = "/home/earthsense/Documents/collection-130624_040630_zed_camera_vis/000003.jpg"
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)

# Global variable to store bounding boxes
bounding_boxes = []

# Function to be called when a bounding box is drawn
def onselect(eclick, erelease):
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    # Create a bounding box in XYXY format
    bounding_boxes.append([x1, y1, x2, y2])
    print(f"Bounding box: ({x1}, {y1}) to ({x2}, {y2})")

# Display the image and enable RectangleSelector
fig, ax = plt.subplots()
ax.imshow(image_np)

# RectangleSelector for drawing bounding boxes
rect_selector = RectangleSelector(ax, onselect, drawtype='box', useblit=True,
                                  button=[1], minspanx=5, minspany=5, spancoords='pixels',
                                  interactive=True)

plt.show()