import cv2
import numpy as np
import argparse

# Argument parser for image path
parser = argparse.ArgumentParser(description='Homography Demo with OpenCV')
parser.add_argument('image_path', type=str, help='Path to the input image')
args = parser.parse_args()

# Globals for storing points
src_points = []
dst_points = []
collecting_src = True

def click_event(event, x, y, flags, param):
    global src_points, dst_points, collecting_src
    img = param['img']
    display = param['display']
    if collecting_src and event == cv2.EVENT_LBUTTONDOWN and len(src_points) < 4:
        src_points.append([x, y])
        cv2.circle(display, (x, y), 5, (0, 255, 0), -1)  # Green for src
        cv2.imshow('Select Points', display)
        if len(src_points) == 4:
            collecting_src = False
            print("Now select 4 destination points (red).")
    elif not collecting_src and event == cv2.EVENT_LBUTTONDOWN and len(dst_points) < 4:
        dst_points.append([x, y])
        cv2.circle(display, (x, y), 5, (0, 0, 255), -1)  # Red for dst
        cv2.imshow('Select Points', display)

def main():
    global src_points, dst_points, collecting_src

    # Load image
    img = cv2.imread(args.image_path)
    if img is None:
        print("Error: Image not found.")
        return

    img_copy = img.copy()
    display = img.copy()
    print("Select 4 source points (green) on the image.")
    cv2.imshow('Select Points', display)
    cv2.setMouseCallback('Select Points', click_event, {'img': img_copy, 'display': display})

    # Wait until 8 points are selected
    while len(src_points) < 4 or len(dst_points) < 4:
        cv2.waitKey(1)
    cv2.destroyWindow('Select Points')

    # Compute homography
    src_pts = np.array(src_points, dtype=np.float32)
    dst_pts = np.array(dst_points, dtype=np.float32)
    H, status = cv2.findHomography(src_pts, dst_pts)

    # Warp perspective
    warped = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))

    # Show result
    cv2.imshow('Warped Image', warped)
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()