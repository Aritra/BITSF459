

# BITS F459 Computer Vision Lab 6 — Optical Flow and Tracking

## 📌 Objective

The purpose of this lab is to understand how **optical flow** can be used for motion estimation and tracking, and how **tracking methods** (KLT vs template matching) work. We will analyze the provided codes and then extend them with more advanced tasks.

---

## 📂 Part 1 — Optical Flow (`optical_flow.py`)

This file demonstrates **dense and sparse optical flow visualization** using OpenCV.

### Function Explanations

* **`draw_dense_flow(flow)`**

  * Takes an optical flow field and visualizes it as a color-coded image in HSV space.
  * Hue = flow direction, Value = flow magnitude.

* **`draw_flow(img, flow, step=16)`**

  * Draws arrows on a grid over the image to represent optical flow vectors.
  * Each arrow points in the direction of motion at that location.
  * The color of the arrow is based on flow direction.

### Main Loop

* Captures webcam feed.
* Press **SPACE**: Start/stop optical flow visualization.
* Press **D**: Toggle between dense and sparse visualization.
* Press **ESC**: Exit.
* Uses **Farneback’s dense optical flow algorithm**:

  ```python
  flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                      pyr_scale=0.25, levels=3, winsize=15,
                                      iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
  ```

---

## 📂 Part 2 — Tracking ROI (`tracking_roi.py`)

This file shows **ROI tracking** using either **KLT optical flow** or **Template Matching**.

### Function Explanations

* **`template_matching_shift(roi_img, search_img)`**

  * Uses normalized cross-correlation to find where the ROI matches best in the search window.
  * Returns the shift `(dx, dy)` of the ROI.

* **`klt_shift(prev_roi, curr_roi)`**

  * Uses **Shi-Tomasi corner detection** (`cv2.goodFeaturesToTrack`) to find keypoints.
  * Tracks them between frames using **Lucas-Kanade Optical Flow** (`cv2.calcOpticalFlowPyrLK`).
  * Computes the average displacement to estimate ROI movement.

* **`mouse_callback(event, x, y, flags, param)`**

  * Allows the user to **draw/select ROI** with the mouse.
  * Left-click and drag → select ROI.

### Main Loop

* Webcam feed captured with `cv2.VideoCapture(0)`.
* If ROI selected:

  * Tracks ROI with either **KLT** (if `USE_KLT=True`) or **Template Matching**.
  * Updates ROI position based on estimated shift.
* Controls:

  * **T**: Toggle tracking.
  * **R**: Reset ROI.
  * **ESC**: Exit.

---

## 📝 Student Tasks

### **Task 1: Background Replacement using Optical Flow**

* Use clustering (e.g., **K-means** or **DBSCAN**) on flow magnitudes to separate:

  * **Static background** (low flow).
  * **Moving foreground** (high flow).
* Replace the static background with an image of your choice (like Zoom/Google Meet virtual background).

💡 *Hint:* Cluster `flow_magnitudes = np.sqrt(fx**2 + fy**2)`.

---

### **Task 2: Improving the Tracker**

* The current tracker sometimes **loses the ROI** or **switches targets**.
* Suggested improvements:

  * Re-initialize features when the number of valid points drops below a threshold.
  * Expand/shrink ROI adaptively based on object motion.
  * Use a confidence measure to decide when to reset.

---

### **Task 3: Multi-Object Tracking**

* Extend the tracker so multiple ROIs can be selected and tracked simultaneously.
* Suggested user interface:

  * Allow multiple ROI selections using repeated **mouse drags**.
  * Track each ROI independently with either KLT or Template Matching.
  * Display bounding boxes in different colors.

---

## ✅ Deliverables

1. Modified `optical_flow.py` with background replacement.
2. Modified `tracking_roi.py` with improved single-object tracking.
3. Extended `tracking_roi.py` (or new file) with multi-object tracking.
4. A short demo video showing your implementations in action.

---
