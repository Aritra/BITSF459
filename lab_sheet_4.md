# Lab Sheet 4: Camera Calibration using Chessboard Method

---

## Objective
In this lab, you will learn how to perform **camera calibration** using the **chessboard method**, understand the **camera intrinsic matrix**, and apply calibration to remove distortion.  
Finally, you will explore a simple **structure-from-motion (SfM)** pipeline that generates a sparse 3D point cloud from your webcam images.

---

## Background

### Steps of Camera Calibration using Chessboard
1. **Calibration pattern**  
   - A chessboard with known square size is used as the reference pattern.  
   - Each inner corner of the chessboard provides a precise 2D reference point.

2. **Capture multiple images**  
   - Take several snapshots of the chessboard from different angles and positions.  
   - These provide correspondences between known **3D world coordinates** (on the flat chessboard plane) and their **2D image projections**.

3. **Corner detection**  
   - The algorithm detects chessboard corners in each snapshot.  
   - Sub-pixel refinement increases accuracy.

4. **Estimate parameters**  
   - Using multiple sets of correspondences, the calibration algorithm estimates:  
     - **Intrinsic matrix (K)**  
     - **Distortion coefficients**  
     - **Extrinsics (R, t)** for each view.

---

### The Camera Intrinsic Matrix
The intrinsic matrix \(K\) is:

$$
K =
\begin{bmatrix}
f_x & s & c_x \\
0   & f_y & c_y \\
0   & 0   & 1
\end{bmatrix}
$$

- $$\(f_x, f_y\)$$: focal lengths in pixel units.  
- $$\(c_x, c_y\)$$: principal point (optical center).  
- $$\(s\)$$: skew factor (often zero).  

**Physical meaning:**  
- Focal lengths determine field of view.  
- Principal point tells where the optical axis meets the image plane.  
- Skew indicates whether pixel axes are perpendicular (usually they are).

---

## Instructions

### Part A: Camera Calibration
- Run **`camera_calibrator.py`**.  
- Press **SPACE** to take snapshots of the chessboard.  
- When you have at least 3 snapshots, press **d** to perform calibration.  
- The program computes and prints the **intrinsic matrix (K)** and saves it as `cam_K.npy`.  
- It also shows the original vs. undistorted chessboard corners for visual confirmation.

---

### Part B: Undistortion Test
- Run **`undistort.py`**.  
- This script uses your calibration (`cam_K.npy`) to undistort the webcam feed.  
- A **red ROI rectangle** and **green lines** are drawn to test rectification.  
- **Good calibration test:** straight lines in the real world appear **straight** in the undistorted feed, even at the image edges.

---

### Part C: Sparse 3D Reconstruction
- Open **`single_cam_sfm.py`**.  
- Workflow:
  1. Run the script.  
  2. Press **SPACE** to capture multiple snapshots of your face or an object.  
  3. Press **d** to process images.  
  4. The script extracts **SIFT features**, matches them, estimates motion using the **Essential Matrix**, and triangulates 3D points.  
  5. A sparse 3D point cloud is saved as **`3d_pc.ply`**.  

- Load the `.ply` file in a 3D viewer (MeshLab, CloudCompare, etc.) to inspect your sparse blob point cloud.

---

## Deliverables
1. Camera intrinsic matrix \(K\) (from calibration).  
2. Screenshot or description of undistortion test results.  
3. Sparse 3D point cloud file (`.ply`) generated from `single_cam_sfm.py`.

---

## Notes
- At least **3 snapshots** are required for calibration.  
- Calibration accuracy improves with more views at different angles.  
- For SfM, ensure **sufficient texture** in the object to get reliable feature matches.  
- `single_cam_sfm.py` may need debugging if triangulation or matching fails — fixing these issues is part of the exercise.

---

## Appendix: Explanation of Key OpenCV Functions

### 1. `cv2.findChessboardCorners(gray, (CHESSBOARD_COLS, CHESSBOARD_ROWS), None)`
- **Purpose:** Detects the positions of internal chessboard corners in a grayscale image.  
- **Inputs:**  
  - `gray`: Grayscale input image.  
  - `(CHESSBOARD_COLS, CHESSBOARD_ROWS)`: Number of inner corners per row and column of the chessboard pattern.  
- **Output:**  
  - Returns a boolean flag (found or not) and an array of detected corner points.  
- **Use:** First step in calibration to locate 2D feature points.

---

### 2. `cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)`
- **Purpose:** Refines the detected corner positions to **sub-pixel accuracy**.  
- **Inputs:**  
  - `gray`: Grayscale image.  
  - `corners`: Initial corner estimates from `findChessboardCorners`.  
  - `(11,11)`: Search window size for refinement.  
  - `(-1,-1)`: No dead zone around corners.  
  - `criteria`: Stopping criteria (e.g., max iterations or minimum error).  
- **Output:** Updated corner positions with higher precision.  
- **Use:** Improves accuracy of calibration.

---

### 3. `cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)`
- **Purpose:** Estimates the **camera intrinsic matrix**, distortion coefficients, and extrinsic parameters.  
- **Inputs:**  
  - `objpoints`: 3D world points (e.g., chessboard corners in real-world coordinates).  
  - `imgpoints`: Corresponding 2D points in images.  
  - `gray.shape[::-1]`: Image size (width, height).  
- **Output:**  
  - RMS reprojection error.  
  - Intrinsic matrix `K`.  
  - Distortion coefficients.  
  - Rotation and translation vectors for each view.  
- **Use:** Core step to obtain camera calibration parameters.

---

### 4. `cv2.undistort(img, mtx, dist, None, mtx)`
- **Purpose:** Removes lens distortion from an image using calibration results.  
- **Inputs:**  
  - `img`: Distorted input image.  
  - `mtx`: Intrinsic matrix.  
  - `dist`: Distortion coefficients.  
- **Output:** Undistorted image.  
- **Use:** Verification of calibration — straight lines remain straight.

---

### 5. `cv2.getOptimalNewCameraMatrix(cam_K, dist_coeffs, (w, h), 1, (w, h))`
- **Purpose:** Computes a new camera matrix that adjusts the field of view while minimizing black regions after undistortion.  
- **Inputs:**  
  - `cam_K`: Original intrinsic matrix.  
  - `dist_coeffs`: Distortion coefficients.  
  - `(w,h)`: Image width and height.  
  - `1`: Free scaling parameter (1 keeps all pixels, 0 crops).  
- **Output:** New camera matrix and region of interest.  
- **Use:** Optimizes undistortion for practical use.

---

### 6. `cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)`
- **Purpose:** Reconstructs 3D points from two camera views.  
- **Inputs:**  
  - `P1, P2`: Projection matrices for the two views.  
  - `pts1, pts2`: Matched 2D feature points (transposed to shape 2×N).  
- **Output:** Homogeneous 4×N matrix of reconstructed 3D points.  
- **Use:** Basis for structure-from-motion and 3D reconstruction.

---

