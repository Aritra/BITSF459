import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

def show_color_components(img, color_space, titles, as_gray=False, fig=None, row=0):
    """Splits and displays the color components of an image."""
    channels = cv2.split(img)
    for i, (channel, title) in enumerate(zip(channels, titles)):
        ax = fig.add_subplot(3, 3, row*3 + i + 1)
        if as_gray:
            ax.imshow(channel, cmap='gray')
        else:
            ax.imshow(channel, cmap='gray' if color_space != 'RGB' else None)
        ax.set_title(f"{color_space} - {title}")
        ax.axis('off')

def main():
    parser = argparse.ArgumentParser(description='Show RGB, HSV, LAB components of an image.')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    args = parser.parse_args()

    # Read image in BGR
    img_bgr = cv2.imread(args.image_path)
    if img_bgr is None:
        print("Error: Image not found or unable to read.")
        return

    # Convert to RGB for display
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    # Prepare a single figure for all components
    fig = plt.figure(figsize=(12, 12))

    # RGB components as grayscale
    show_color_components(img_rgb, 'RGB', ['Red', 'Green', 'Blue'], as_gray=True, fig=fig, row=0)
    # HSV components
    show_color_components(img_hsv, 'HSV', ['Hue', 'Saturation', 'Value'], as_gray=True, fig=fig, row=1)
    # LAB components
    show_color_components(img_lab, 'LAB', ['L*', 'a*', 'b*'], as_gray=True, fig=fig, row=2)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()