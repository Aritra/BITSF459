import cv2
import numpy as np

def print_help():
    print("""
This program applies scaling, translation, and rotation to an image using OpenCV.
You will be prompted for:
- Image path
- Scaling factors (sx, sy)
- Translation values (tx, ty)
- Rotation angle (in degrees)

Transformations are applied in this order: scaling -> translation -> rotation.
""")

def main():
    print_help()
    img_path = input("Enter image path: ")
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Could not load image.")
        return

    try:
        sx = float(input("Enter scaling factor sx (e.g. 1.2): "))
        sy = float(input("Enter scaling factor sy (e.g. 1.2): "))
        tx = float(input("Enter translation tx (pixels): "))
        ty = float(input("Enter translation ty (pixels): "))
        angle = float(input("Enter rotation angle (degrees): "))
    except ValueError:
        print("Invalid input.")
        return

    # Scaling
    scaled = cv2.resize(img, None, fx=sx, fy=sy, interpolation=cv2.INTER_LINEAR)

    # Translation
    rows, cols = scaled.shape[:2]
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(scaled, M_trans, (cols, rows))

    # Rotation
    center = (cols // 2, rows // 2)
    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(translated, M_rot, (cols, rows))

    # Display images
    cv2.imshow('Input Image', img)
    cv2.imshow('Transformed Image', rotated)
    print("Press any key in the image window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()