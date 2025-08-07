import cv2
import numpy as np
import argparse
import sys
import math

def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply affine shearing transformation to an image using OpenCV."
    )
    parser.add_argument(
        "--image", "-i", required=True, type=str,
        help="Path to the input image."
    )
    parser.add_argument(
        "--x_angle", "-x", required=False, type=float, default=0.0,
        help="Shearing angle along the x-axis in degrees (default: 0)."
    )
    parser.add_argument(
        "--y_angle", "-y", required=False, type=float, default=0.0,
        help="Shearing angle along the y-axis in degrees (default: 0)."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Load image
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not load image at '{args.image}'")
        sys.exit(1)

    rows, cols = img.shape[:2]

    # Convert angles to radians and calculate shear factors
    shear_x = math.tan(math.radians(args.x_angle))
    shear_y = math.tan(math.radians(args.y_angle))

    # Affine transformation matrix for shearing
    M = np.float32([
        [1, shear_x, 0],
        [shear_y, 1, 0]
    ])

    # Calculate new image size to fit the sheared image
    new_cols = int(cols + abs(shear_x) * rows)
    new_rows = int(rows + abs(shear_y) * cols)

    sheared_img = cv2.warpAffine(img, M, (new_cols, new_rows), borderMode=cv2.BORDER_REPLICATE)

    # Show and save result
    cv2.imshow("Original Image", img)
    cv2.imshow("Sheared Image", sheared_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("Use --help for usage information.")
        sys.exit(1)