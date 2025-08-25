import cv2
import numpy as np

def draw_orb_keypoints(frame, keypoints):
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        angle = np.deg2rad(kp.angle)
        length = int(kp.response * 20)  # Scale confidence for visualization
        dx = int(length * np.cos(angle))
        dy = int(length * np.sin(angle))
        end_point = (x + dx, y + dy)
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        cv2.line(frame, (x, y), end_point, (0, 0, 255), 2)
    return frame

def main():
    cap = cv2.VideoCapture(0)
    orb = cv2.ORB_create()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints = orb.detect(gray, None)

        frame_with_kp = draw_orb_keypoints(frame, keypoints)

        cv2.imshow('ORB Features', frame_with_kp)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()