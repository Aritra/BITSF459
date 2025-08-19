import cv2
import numpy as np

# Hardcoded image path
IMAGE_PATH = '/home/aritra/CV_codebase/table_top.png'

def nothing(x):
    pass

def apply_morph(mask, op):
    kernel = np.ones((3, 3), np.uint8)
    if op == 'dilate':
        return cv2.dilate(mask, kernel, iterations=1)
    elif op == 'erode':
        return cv2.erode(mask, kernel, iterations=1)
    return mask

def morph_window(mask):
    temp_mask = mask.copy()
    cv2.namedWindow('Morph Operations')
    while True:
        display = temp_mask.copy()
        # Draw buttons
        cv2.rectangle(display, (10, 10), (110, 60), (200, 200, 200), -1)
        cv2.putText(display, 'Dilate', (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        cv2.rectangle(display, (130, 10), (230, 60), (200, 200, 200), -1)
        cv2.putText(display, 'Erode', (140, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        cv2.imshow('Morph Operations', display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break
        # Mouse click handling
        def on_mouse(event, x, y, flags, param):
            nonlocal temp_mask
            if event == cv2.EVENT_LBUTTONDOWN:
                if 10 < x < 110 and 10 < y < 60:
                    temp_mask = apply_morph(temp_mask, 'dilate')
                elif 130 < x < 230 and 10 < y < 60:
                    temp_mask = apply_morph(temp_mask, 'erode')
        cv2.setMouseCallback('Morph Operations', on_mouse)
    cv2.destroyWindow('Morph Operations')

def main():
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image not found!")
        return

    cv2.namedWindow('Input')
    cv2.namedWindow('Mask')

    # Trackbar for threshold
    cv2.createTrackbar('Threshold', 'Input', 127, 255, nothing)

    mask_btn_pos = (10, 10, 110, 60)  # x1, y1, x2, y2

    while True:
        thresh_val = cv2.getTrackbarPos('Threshold', 'Input')
        _, mask = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)
        cv2.imshow('Input', img)

        # Draw button on mask window
        mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(mask_display, (mask_btn_pos[0], mask_btn_pos[1]), (mask_btn_pos[2], mask_btn_pos[3]), (200,200,200), -1)
        cv2.putText(mask_display, 'Morph', (mask_btn_pos[0]+10, mask_btn_pos[1]+35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        cv2.imshow('Mask', mask_display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break

        # Mouse click handling for mask window
        def on_mask_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if mask_btn_pos[0] < x < mask_btn_pos[2] and mask_btn_pos[1] < y < mask_btn_pos[3]:
                    morph_window(mask)
        cv2.setMouseCallback('Mask', on_mask_mouse)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()