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
