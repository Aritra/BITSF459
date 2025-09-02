import cv2
import numpy as np
import os

# Load camera matrix
cam_K_path = os.path.join(os.path.dirname(__file__), 'cam_K.npy')
cam_K = np.load(cam_K_path)

# Assume zero distortion for pinhole model estimation
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()
    
ret, frame = cap.read()
if not ret:
    exit("Failed to read from webcam")
else:
    h, w = frame.shape[:2]
 # Get optimal new camera matrix and ROI
    new_cam_K, roi = cv2.getOptimalNewCameraMatrix(cam_K, dist_coeffs, (w, h), 1, (w, h))
    
print("Optimal new camera matrix:  ")
print(new_cam_K)
print("ROI:  ")
print(roi)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

   
    # Undistort frame
    undistorted = cv2.undistort(frame, cam_K, dist_coeffs, None, new_cam_K)

    # ROI: x, y, w, h
    x, y, rw, rh = roi

    # Draw red rectangle on original frame
    frame_rect = frame.copy()
    cv2.rectangle(frame_rect, (x, y), (x + rw, y + rh), (0, 0, 255), 2)

    # Draw green parallel lines inside the rectangle
    num_lines = 5
    for i in range(1, num_lines):
        y_line = y + i * rh // num_lines
        cv2.line(frame_rect, (x, y_line), (x + rw, y_line), (0, 255, 0), 1)

    # Show original frame with ROI
    cv2.imshow('Original with ROI', frame_rect)
    # Show undistorted ROI only
    undistorted_roi = undistorted[y:y+rh, x:x+rw]
    cv2.imshow('Undistorted ROI', undistorted_roi)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()