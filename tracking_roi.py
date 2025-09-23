import cv2
import numpy as np

# --- Tracking method flag ---
USE_KLT = True  # Set to True for KLT, False for template matching

def template_matching_shift(roi_img, search_img):
    """Returns (dx, dy) shift of ROI in search_img using template matching."""
    cv2.imshow("ROI", roi_img)
    cv2.imshow("Search Window", search_img)
    cv2.waitKey(10)
    res = cv2.matchTemplate(search_img, roi_img, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    dx = max_loc[0] - (search_img.shape[1] // 2 - roi_img.shape[1] // 2)
    dy = max_loc[1] - (search_img.shape[0] // 2 - roi_img.shape[0] // 2)
    print(f"Template matching shift: dx={dx}, dy={dy}")
    return dx, dy

def klt_shift(prev_roi, curr_roi):
    """Returns (dx, dy) shift of ROI between prev_roi and curr_roi using Shi-Tomasi + KLT optical flow. Also draws matches."""
    prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=20, qualityLevel=0.01, minDistance=5)
    if corners is None:
        return 0, 0
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, corners, None)
    if next_pts is None or status is None:
        return 0, 0
    valid = status.flatten() == 1
    if not np.any(valid):
        return 0, 0
    shift = (next_pts[valid] - corners[valid]).mean(axis=0)
    print("shift: "+str(shift))
    dx, dy = int(shift[0][0]), int(shift[0][1])
    # Draw matches
    match_img = np.hstack([prev_roi, curr_roi])
    for i, v in enumerate(valid):
        if v:
            pt1 = tuple(np.round(corners[i,0]).astype(int))
            pt2 = tuple(np.round(next_pts[i,0]).astype(int) + np.array([prev_roi.shape[1],0]))
            cv2.circle(match_img, pt1, 3, (0,255,0), -1)
            cv2.circle(match_img, pt2, 3, (0,0,255), -1)
            cv2.line(match_img, pt1, pt2, (255,0,0), 1)
    cv2.imshow("KLT Matches", match_img)
    return dx, dy

drawing = False
ix, iy = -1, -1
rx, ry, rw, rh = -1, -1, -1, -1
roi_selected = False
tracking = False
roi_img = None
last_roi = None  # For KLT
last_frame = None  # For KLT

def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, rx, ry, rw, rh, roi_selected, tracking
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        roi_selected = False
        tracking = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            rx, ry = min(ix, x), min(iy, y)
            rw, rh = abs(ix - x), abs(iy - y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rx, ry = min(ix, x), min(iy, y)
        rw, rh = abs(ix - x), abs(iy - y)
        if rw > 10 and rh > 10:
            roi_selected = True

cap = cv2.VideoCapture(0)
cv2.namedWindow('Tracking ROI')
cv2.setMouseCallback('Tracking ROI', mouse_callback)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    display = frame.copy()

    # Draw ROI if being drawn or selected
    if drawing or roi_selected:
        if rw > 0 and rh > 0:
            color = (0,0,255) if USE_KLT else (0,255,0)
            cv2.rectangle(display, (rx, ry), (rx+rw, ry+rh), color, 2)

    if roi_selected:
        if tracking:
            if roi_img is None:
                roi_img = frame[ry:ry+rh, rx:rx+rw].copy()
            if USE_KLT:
                if last_roi is not None:
                    curr_roi = frame[ry:ry+rh, rx:rx+rw].copy()
                    dx, dy = klt_shift(last_roi, curr_roi)
                    rx = max(0, min(frame.shape[1]-rw, rx+dx))
                    ry = max(0, min(frame.shape[0]-rh, ry+dy))
                    last_roi = frame[ry:ry+rh, rx:rx+rw].copy()
                else:
                    last_roi = roi_img.copy()
            else:
                cx, cy = rx + rw//2, ry + rh//2
                sw = min(frame.shape[1], rw*3)
                sh = min(frame.shape[0], rh*3)
                sx = max(0, cx - sw//2)
                sy = max(0, cy - sh//2)
                sx = min(sx, frame.shape[1] - sw)
                sy = min(sy, frame.shape[0] - sh)
                sx, sy = int(sx), int(sy)
                sw, sh = int(sw), int(sh)
                cv2.rectangle(display, (sx, sy), (sx+sw, sy+sh), (0,0,255), 2)
                cv2.rectangle(display, (rx, ry), (rx+rw, ry+rh), (0,255,0), 2)
                search_img = frame[sy:sy+sh, sx:sx+sw]
                dx, dy = template_matching_shift(roi_img, search_img)
                cx_new = cx + dx
                cy_new = cy + dy
                rx = max(0, min(frame.shape[1]-rw, cx_new - rw//2))
                ry = max(0, min(frame.shape[0]-rh, cy_new - rh//2))
        else:
            roi_img = None
            last_roi = None

    cv2.imshow('Tracking ROI', display)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break
    elif key == ord('t') and roi_selected:
        tracking = not tracking
        if not tracking:
            roi_img = None
            last_roi = None
    elif key == ord('r'):
        roi_selected = False
        tracking = False
        rx, ry, rw, rh = -1, -1, -1, -1
        roi_img = None
        last_roi = None

cap.release()
cv2.destroyAllWindows()