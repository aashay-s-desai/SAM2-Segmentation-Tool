import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Load the image
image_path = "/home/earthsense/Documents/collection-130624_040630_zed_camera_vis/000003.jpg"
image = Image.open(image_path).convert("RGB")

# Convert to numpy array for display
image_np = np.array(image)

# Function to capture clicks
def onclick(event):
    x = int(event.xdata)
    y = int(event.ydata)
    print(f"Point: ({x}, {y})")

# Display the image
fig, ax = plt.subplots()
ax.imshow(image_np)
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
