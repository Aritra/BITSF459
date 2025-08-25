import cv2
import numpy as np
from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt

# Hardcoded image path
IMG_PATH = '/home/aritra/CV_codebase/STI/Old_classic/cameraman.pgm'  # Change to your image path

def show_gaussian_octaves(img, num_octaves=4, num_scales=5):
    fig, axes = plt.subplots(num_octaves, num_scales, figsize=(12, 8))
    octaves = []
    for o in range(num_octaves):
        octave_imgs = []
        base = cv2.resize(img, (img.shape[1] // (2 ** o), img.shape[0] // (2 ** o)))
        for s in range(num_scales):
            sigma = 1.6 * (2 ** (s / num_scales))
            blur = cv2.GaussianBlur(base, (0, 0), sigmaX=sigma, sigmaY=sigma)
            axes[o, s].imshow(blur, cmap='gray')
            axes[o, s].set_title(f'O{o+1} S{s+1} σ={sigma:.2f}')
            axes[o, s].axis('off')
            octave_imgs.append(blur)
        octaves.append(octave_imgs)
    plt.suptitle('Gaussian Octaves and Scales')
    plt.tight_layout()
    plt.show()
    return octaves

def show_dog_octaves(octaves):
    fig, axes = plt.subplots(len(octaves), len(octaves[0])-1, figsize=(12, 8))
    dogs = []
    for o, octave_imgs in enumerate(octaves):
        dog_imgs = []
        for s in range(len(octave_imgs)-1):
            dog = cv2.subtract(octave_imgs[s+1], octave_imgs[s])
            axes[o, s].imshow(dog, cmap='gray')
            axes[o, s].set_title(f'O{o+1} DOG{s+1}')
            axes[o, s].axis('off')
            dog_imgs.append(dog)
        dogs.append(dog_imgs)
    plt.suptitle('Difference of Gaussian (DoG)')
    plt.tight_layout()
    plt.show()
    return dogs

def draw_keypoints(img, keypoints, title='Keypoints'):
    img_kp = cv2.drawKeypoints(img, keypoints, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(8, 8))
    plt.imshow(img_kp, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def draw_keypoints_on_dog(dog, keypoints, title='Keypoints on DoG'):
    img_kp = cv2.drawKeypoints(dog, keypoints, None, color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(8, 8))
    plt.imshow(img_kp, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def get_patch(img, kp, size=16):
    x, y = int(kp.pt[0]), int(kp.pt[1])
    half = size // 2
    patch = img[max(0, y-half):y+half, max(0, x-half):x+half]
    return patch

def show_descriptor_histogram(patch):
    if patch.shape[0] < 16 or patch.shape[1] < 16:
        patch = cv2.copyMakeBorder(patch, 0, 16-patch.shape[0], 0, 16-patch.shape[1], cv2.BORDER_CONSTANT, value=0)
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hist_blocks = []
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(4):
        for j in range(4):
            block_mag = mag[i*4:(i+1)*4, j*4:(j+1)*4].flatten()
            block_ang = ang[i*4:(i+1)*4, j*4:(j+1)*4].flatten()
            hist, bins = np.histogram(block_ang, bins=8, range=(0, 360), weights=block_mag)
            axes[i, j].bar(np.arange(8), hist)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].set_title(f'Block {i*4+j+1}')
    plt.suptitle('Histogram of Gradients in 4x4 blocks (SIFT Descriptor)')
    plt.tight_layout()
    plt.show()

def main():
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image not found!")
        return

    # Step 1: Gaussian Octaves
    octaves = show_gaussian_octaves(img)
    input("Press [q] then Enter to continue to DoG...")

    # Step 2: DoG
    dogs = show_dog_octaves(octaves)
    input("Press [q] then Enter to continue to raw keypoints...")

    # Step 3: Raw Keypoints
    sift = cv2.SIFT_create(contrastThreshold=0.01, edgeThreshold=10)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    draw_keypoints_on_dog(dogs[0][1], keypoints, title='Raw Keypoints on DoG')
    input("Press [q] then Enter to continue to refined keypoints...")

    # Step 4: Refined Keypoints (SIFT does refinement internally)
    draw_keypoints(img, keypoints, title='Refined Keypoints on Image')

    # Step 5: Click to show descriptor
    print("Click on a keypoint in the window to see its descriptor histogram.")
    fig, ax = plt.subplots(figsize=(8, 8))
    img_kp = cv2.drawKeypoints(img, keypoints, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    ax.imshow(img_kp, cmap='gray')
    ax.set_title('Click on a keypoint')
    ax.axis('off')

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        # Find nearest keypoint
        min_dist = float('inf')
        nearest_kp = None
        for kp in keypoints:
            dist = np.hypot(kp.pt[0] - x, kp.pt[1] - y)
            if dist < min_dist:
                min_dist = dist
                nearest_kp = kp
        if nearest_kp and min_dist < 20:
            patch = get_patch(img, nearest_kp, size=16)
            show_descriptor_histogram(patch)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

if __name__ == "__main__":
    main()