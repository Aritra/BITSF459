import cv2
import numpy as np
import os

# Chessboard parameters
CHESSBOARD_ROWS = 6
CHESSBOARD_COLS = 9
SQUARE_SIZE_MM = 40  # millimeters

# Termination criteria for cornerSubPix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), ..., (9,6,0) scaled by square size
objp = np.zeros((CHESSBOARD_ROWS * CHESSBOARD_COLS, 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_COLS, 0:CHESSBOARD_ROWS].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane
snapshots = []
snapshot_count = 0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press SPACE to take a snapshot. Press 'd' to calibrate and display results. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    cv2.imshow('Camera', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # Take snapshot
        snapshots.append(frame.copy())
        snapshot_count += 1
        print(f"snapshot...{snapshot_count}")

    elif key == ord('d'):  # Calibrate
        if len(snapshots) < 3:
            print("Need at least 3 snapshots for calibration.")
            continue

        objpoints.clear()
        imgpoints.clear()
        print("Detecting corners and calibrating...")

        for idx, img in enumerate(snapshots):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_COLS, CHESSBOARD_ROWS), None)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2)
            else:
                print(f"Chessboard not detected in snapshot {idx+1}")

        if len(objpoints) < 3:
            print("Not enough valid snapshots with detected chessboard corners.")
            continue

        # Camera calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print("Camera calibration matrix (K):")
        print(mtx)
        np.save("cam_K.npy", mtx)
        print("Saved camera matrix as cam_K.npy")

        # Show results for each snapshot
        for idx, img in enumerate(snapshots):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_COLS, CHESSBOARD_ROWS), None)
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                img_corners = cv2.drawChessboardCorners(img.copy(), (CHESSBOARD_COLS, CHESSBOARD_ROWS), corners2, ret)

                # Undistort image
                undistorted = cv2.undistort(img, mtx, dist, None, mtx)
                gray_undist = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
                ret_undist, corners_undist = cv2.findChessboardCorners(gray_undist, (CHESSBOARD_COLS, CHESSBOARD_ROWS), None)
                if ret_undist:
                    corners2_undist = cv2.cornerSubPix(gray_undist, corners_undist, (11,11), (-1,-1), criteria)
                    undist_corners = cv2.drawChessboardCorners(undistorted.copy(), (CHESSBOARD_COLS, CHESSBOARD_ROWS), corners2_undist, ret_undist)
                else:
                    undist_corners = undistorted.copy()

                cv2.imshow(f"Original with corners {idx+1}", img_corners)
                cv2.imshow(f"Rectified with corners {idx+1}", undist_corners)
                cv2.waitKey(0)
                cv2.destroyWindow(f"Original with corners {idx+1}")
                cv2.destroyWindow(f"Rectified with corners {idx+1}")
            else:
                print(f"Chessboard not detected in snapshot {idx+1}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()