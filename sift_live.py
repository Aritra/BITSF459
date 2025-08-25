import cv2
import numpy as np

def draw_keypoints_with_orientation(img, keypoints, confidences):
    for kp, conf in zip(keypoints, confidences):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        angle = kp.angle
        length = int(20 * conf)  # scale length by confidence
        color = (0, int(255 * conf), int(255 * (1 - conf)))  # green for high, red for low
        # Draw circle for keypoint
        cv2.circle(img, (x, y), 3, color, -1)
        # Draw orientation arrow
        dx = int(length * np.cos(np.deg2rad(angle)))
        dy = int(length * np.sin(np.deg2rad(angle)))
        cv2.arrowedLine(img, (x, y), (x + dx, y + dy), color, 2, tipLength=0.3)
        # Draw confidence as text
        cv2.putText(img, f"{conf:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow('SIFT Features')
cv2.createTrackbar('Top %', 'SIFT Features', 40, 40, nothing)  # 40% to 10%

sift = cv2.SIFT_create()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is not None and len(keypoints) > 0:
        # Compute confidence as norm of descriptor (L2 norm)
        norms = np.linalg.norm(descriptors, axis=1)
        # Normalize to [0,1]
        confs = (norms - norms.min()) / (norms.ptp() + 1e-8)
        # Sort keypoints by confidence
        sorted_idx = np.argsort(-confs)
        keypoints = [keypoints[i] for i in sorted_idx]
        confs = confs[sorted_idx]

        # Get trackbar value
        top_percent = cv2.getTrackbarPos('Top %', 'SIFT Features')
        top_percent = max(10, min(top_percent, 40))
        n_show = max(1, int(len(keypoints) * top_percent / 100))
        keypoints_show = keypoints[:n_show]
        confs_show = confs[:n_show]

        draw_keypoints_with_orientation(frame, keypoints_show, confs_show)

    cv2.imshow('SIFT Features', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()