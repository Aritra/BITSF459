import cv2
import numpy as np
from scipy import ndimage

import matplotlib.pyplot as plt

# Hardcoded image path
image_path = '/home/aritra/CV_codebase/STI/Old_classic/cameraman.pgm'

# Read image as grayscale
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 1. Original image
original = img

# 2. Sobel X gradient
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)

# 3. Sobel Y gradient
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)

# 4. Sobel combined
sobel_combined = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

# 5. Laplacian of Gaussian (LoG)
# First apply Gaussian blur
blurred = cv2.GaussianBlur(img, (3, 3), 0)
# Then apply Laplacian
log = cv2.Laplacian(blurred, cv2.CV_64F)
log = cv2.convertScaleAbs(log)

# 6. Prewitt filter
# Define Prewitt kernels
kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]], dtype=int)
kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=int)
prewittx = ndimage.convolve(img, kernelx)
prewitty = ndimage.convolve(img, kernely)
prewitt = cv2.convertScaleAbs(prewittx) + cv2.convertScaleAbs(prewitty)

# Plotting
titles = ['Original', 'Sobel X', 'Sobel Y', 'Sobel Combined', 'LoG', 'Prewitt']
images = [original, sobelx, sobely, sobel_combined, log, prewitt]

plt.figure(figsize=(12, 6))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()