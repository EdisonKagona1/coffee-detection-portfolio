# create_dummy_image.py
import numpy as np
import cv2

# Define image dimensions and create a black image
height = 480
width = 640
# The shape is (height, width, channels), and the data type is 8-bit unsigned integer
blank_image = np.zeros((height, width, 3), np.uint8)

# Define the output filename
filename = "test_image.jpg"

# Save the image using OpenCV
try:
    cv2.imwrite(filename, blank_image)
    print(f"Successfully created dummy image: '{filename}'")
except Exception as e:
    print(f"Error creating image: {e}")