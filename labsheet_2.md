---
# Computer Vision Lab 2: Edge Detection, Morphology, and Object Tracking

## Objectives
In this lab, students will:
- Explore classical **edge detection** methods and study their differences.
- Understand the impact of **threshold selection in Canny edge detection**.
- Learn how **morphological operations** (erosion and dilation) affect binary masks.
- Apply morphology to improve **hand contour detection** and **polygon approximation**.
- Study the application of **Hough Circle Transform** in a ball tracker.
- Extend the tracker to achieve **3D calibration** and smoother, more consistent tracking.

---

## Pre-requisites
- Complete **Lab 1** and ensure your environment is set up with:
  - `opencv-python`
  - `opencv-contrib-python`
  - `numpy`
  - `matplotlib`
  - `open3d` (for future experiments)
- Clone the lab repository if not already done:

```bash
git clone https://github.com/Aritra/BITSF459
cd BITSF459
````

---

## Part 1: Exploring Edge Detection

Open and run the file:

```bash
python opencv_edge_comparison.py
```

### Task:

1. Compare different edge detectors provided in the script (e.g., **Sobel**, **Laplacian of Gaussian (LoG)**, **Canny**).
2. Note down:

   * How the output differs visually for the same image.
   * Situations where one detector performs better than the others.

---

## Part 2: Studying Canny Thresholds

Run:

```bash
python canny_thres_effect.py
```

### Task:

1. Change the **threshold values** using the trackbars provided in the script.
2. Observe:

   * How low thresholds make edges too noisy.
   * How high thresholds cause loss of important edges.
3. Identify a **balanced threshold range** for clean and meaningful edge maps.

---

## Part 3: Morphology Operations

Open:

```bash
python binarize_morph.py
```

### Task:

1. Study how the script uses **mouse clicks** to interact with the image.
2. Understand how `cv2.erode()` and `cv2.dilate()` are applied.
3. Experiment with the number of iterations of erosion and dilation.

   * Erosion removes small noise but can thin out objects.
   * Dilation expands regions and fills small gaps.

---

## Part 4: Hand Contour Detection and Polygon Approximation

Run:

```bash
python hand_contour.py
```

### Task:

1. Observe how contours are detected from the binary mask.
2. Apply the **erosion and dilation operations** (from Part 3) before contour detection.

   * Does the contour look smoother?
   * Is the polygon approximation more stable?
3. Experiment with different approximation factors in `cv2.approxPolyDP()`.

---

## Part 5: Ball Tracking with Hough Circles

Run:

```bash
python ball_tracker.py
```

### Task:

1. Study how the script uses **Hough Circle Transform (`cv2.HoughCircles`)** to track the ball.
2. Note:

   * How sensitive the circle detection is to parameters (e.g., minimum distance, edge detection thresholds, minimum/maximum radius).
   * Cases where false detections occur.

---

## Part 6: Extension Task (Non-evaluative, Exploratory)

### Goal:

Enhance the ball tracker to support **3D calibration and consistent tracking**.

### Ideas:

* Implement a **calibration step** using a known object or pattern (e.g., a checkerboard).
* Use multiple frames to **average detections** and smooth the tracking.
* Maintain a history buffer of positions to reduce jitter in circle detection.
* (Optional advanced idea) If multiple cameras are available, attempt **stereo calibration** to triangulate the ball’s 3D position.

---

## Deliverables

By the end of this lab, you should:

* Have experimented with **different edge detectors** and threshold values.
* Understood the **impact of morphology** on contour detection.
* Explored how **Hough Circles** are used for tracking.
* Produced a modified ball tracker that demonstrates **improved stability** and attempts **3D calibration**.

---
