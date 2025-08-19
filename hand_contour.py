import cv2
import numpy as np

def nothing(x):
    pass

# Create windows
cv2.namedWindow('Webcam')
cv2.namedWindow('Mask')

# Trackbars for hue thresholds
cv2.createTrackbar('Lower Hue', 'Webcam', 5, 179, nothing)
cv2.createTrackbar('Upper Hue', 'Webcam', 35, 179, nothing)

cap = cv2.VideoCapture(0)

box_size = 200

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    roi_x1 = center_x - box_size
    roi_y1 = center_y - box_size
    roi_x2 = roi_x1 + box_size
    roi_y2 = roi_y1 + box_size

    # Clamp ROI to image boundaries
    roi_x1 = max(0, roi_x1)
    roi_y1 = max(0, roi_y1)
    roi_x2 = min(w, roi_x2)
    roi_y2 = min(h, roi_y2)

    # Draw red rectangle on frame
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)

    # Get ROI
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    
    #apply gaussian blur on roi
    roi = cv2.GaussianBlur(roi, (5, 5), 0)

    # Convert ROI to HSV
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Get trackbar positions
    lower_hue = cv2.getTrackbarPos('Lower Hue', 'Webcam')
    upper_hue = cv2.getTrackbarPos('Upper Hue', 'Webcam')

    # Skin color threshold (tune as needed)
    lower_skin = np.array([lower_hue, 40, 60], dtype=np.uint8)
    upper_skin = np.array([upper_hue, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv_roi, lower_skin, upper_skin)

    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    
    # mask = cv2.dilate(mask, kernel, iterations=2)
    # mask = cv2.erode(mask, kernel, iterations=2)
    

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert mask to BGR for colored drawing
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    if contours:
        # Find largest contour
        cnt = max(contours, key=cv2.contourArea)
        # Draw contour in cyan
        cv2.drawContours(mask_bgr, [cnt], -1, (255, 255, 0), 2)

        # Polygon approximation
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # Draw polygon in green
        cv2.polylines(mask_bgr, [approx], True, (0, 255, 0), 2)

        # Print polygon vertices
        print("Polygon vertices:", approx.reshape(-1, 2))

    # Show windows
    cv2.imshow('Webcam', frame)
    cv2.imshow('Mask', mask_bgr)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()