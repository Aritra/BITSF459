import cv2
import numpy as np

import matplotlib.pyplot as plt

# Hardcoded image path
image_path = '/home/aritra/CV_codebase/STI/Old_classic/sailboat.ppm'  # Change this to your image file

# Read the image in grayscale
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Sobel Edge Detection
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)
sobel = np.uint8(np.clip(sobel, 0, 255))

# Laplacian of Gaussian (LoG)
blurred = cv2.GaussianBlur(img, (3, 3), 0)
log = cv2.Laplacian(blurred, cv2.CV_64F)
log = np.uint8(np.clip(np.absolute(log), 0, 255))

# Canny Edge Detection (default thresholds)
canny = cv2.Canny(img, 100, 200)

# Plotting
titles = ['Original Image', 'Sobel', 'LoG', 'Canny']
images = [img, sobel, log, canny]

plt.figure(figsize=(10, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()