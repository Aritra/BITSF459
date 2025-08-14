import cv2
import numpy as np

# Hardcoded image path
image_path = '/home/aritra/CV_codebase/STI/Additional/houses.pgm'  # Change this to your image path

# Read the image
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Convert to grayscale if necessary
if len(img.shape) == 3:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    gray = img.copy()

# Initial Canny thresholds
lower_thresh = 50
upper_thresh = 150

def nothing(x):
    pass

# Create window for input image (zoomable)
cv2.namedWindow('Input Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Input Image', 800, 600)

# Create window for edge map with trackbars
cv2.namedWindow('Edge Map')
cv2.createTrackbar('Lower', 'Edge Map', lower_thresh, 255, nothing)
cv2.createTrackbar('Upper', 'Edge Map', upper_thresh, 255, nothing)

while True:
    # Show input image
    cv2.imshow('Input Image', img)

    # Get current positions of trackbars
    lower = cv2.getTrackbarPos('Lower', 'Edge Map')
    upper = cv2.getTrackbarPos('Upper', 'Edge Map')

    # Compute Canny edge
    edges = cv2.Canny(gray, lower, upper)

    # Show edge map
    cv2.imshow('Edge Map', edges)

    # Exit on ESC key
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()
