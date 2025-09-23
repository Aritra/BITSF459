import cv2
import numpy as np
show_dense = False

def draw_dense_flow(flow):
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ((ang * 180) / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1)
    fx, fy = flow[y, x].T

    # Compute angle and magnitude
    angles = np.arctan2(fy, fx)
    magnitudes = np.sqrt(fx**2 + fy**2)

    # Normalize angles to [0, 179] for HSV
    hsv_angles = ((angles + np.pi) * (179 / (2 * np.pi))).astype(np.uint8)
    hsv = np.zeros((len(x), 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = hsv_angles
    hsv[:, 0, 1] = 255
    hsv[:, 0, 2] = 255
    colors = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)

    for (x1, y1, dx, dy, color) in zip(x, y, fx, fy, colors):
        cv2.arrowedLine(img, (x1, y1), (int(x1+dx), int(y1+dy)), color.tolist(), 1, tipLength=0.3)
    return img

cap = cv2.VideoCapture(0)
prev_gray = None
tracking = False

print("Press SPACE to start/stop optical flow visualization. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        tracking = not tracking
        prev_gray = gray.copy() if tracking else None
    elif key == ord('d'):
        show_dense = not show_dense

    if tracking and prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            pyr_scale=0.25, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        if show_dense:
            vis = frame.copy()
            vis = draw_dense_flow(flow)
        else:
            vis = frame.copy()
            vis = draw_flow(vis, flow, step=16)
        
        cv2.imshow('Optical Flow', vis)
        prev_gray = gray.copy()
        
    else:
        cv2.imshow('Optical Flow', frame)

cap.release()
cv2.destroyAllWindows()