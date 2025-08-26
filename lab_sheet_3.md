## Computer Vision Lab 3: Feature Detection and Matching

## Objective
This lab introduces feature detection, description, and matching techniques. You will implement Harris corner detection, Shi-Tomasi corner detection, SIFT, ORB, and feature matching using BFMatcher and FLANN. Finally, you will estimate the transformation of a patterned sheet.

## Prerequisites
- OpenCV (with contrib modules for SIFT) if you have not downloaded contrib package already.

Install dependencies:
```bash
pip install opencv-contrib-python
```

---

## 1. Corner Detection

### Harris and Shi-Tomasi Corners
Run the following code to visualize Harris and Shi-Tomasi corners on a webcam feed:

```python
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Harris Corner Detection
    harris_corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    harris_display = frame.copy()
    harris_display[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]
    
    # Shi-Tomasi Corner Detection
    st_corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    st_display = frame.copy()
    if st_corners is not None:
        for corner in st_corners:
            x, y = corner.ravel()
            cv2.circle(st_display, (int(x), int(y)), 3, (0, 255, 0), -1)
    
    # Display results
    cv2.imshow('Harris Corners', harris_display)
    cv2.imshow('Shi-Tomasi Corners', st_display)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**Observation:** Note the differences in the number and accuracy of detected points between the two methods.

---

## 2. Feature Detection with SIFT and ORB

### SIFT Implementation
```python
import cv2
import time

cap = cv2.VideoCapture(0)
sift = cv2.SIFT_create()

prev_time = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp = sift.detect(gray, None)
    sift_frame = cv2.drawKeypoints(frame, kp, None)
    
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(sift_frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('SIFT', sift_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### ORB Implementation
```python
import cv2
import time

cap = cv2.VideoCapture(0)
orb = cv2.ORB_create()

prev_time = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    kp = orb.detect(frame, None)
    kp, des = orb.compute(frame, kp)
    orb_frame = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0))
    
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(orb_frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('ORB', orb_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**Observation:** Compare the FPS and feature quality of SIFT vs. ORB.

---

## 3. Feature Matching

### Snapshot Matching with RANSAC
```python
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
snapshots = []
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' '):
        snapshots.append(frame.copy())
        print(f"Snapshot {len(snapshots)} taken")
    
    if len(snapshots) == 2:
        # Extract features
        kp1, des1 = sift.detectAndCompute(cv2.cvtColor(snapshots[0], cv2.COLOR_BGR2GRAY), None)
        kp2, des2 = sift.detectAndCompute(cv2.cvtColor(snapshots[1], cv2.COLOR_BGR2GRAY), None)
        
        # Match features
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        
        # RANSAC filtering
        if len(good) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()
            
            # Draw matches
            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=None,
                               matchesMask=matches_mask,
                               flags=2)
            result = cv2.drawMatches(snapshots[0], kp1, snapshots[1], kp2, good, None, **draw_params)
            cv2.imshow('Matches', result)
        else:
            print("Not enough matches")
            snapshots = []  # Reset
        
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Tasks

### Task 1: Feature Detector and Matcher Comparison
Test combinations of:
- **Detectors:** SIFT, ORB, AKAZE
- **Matchers:** BFMatcher, FLANN

**Instructions:**
1. Modify the snapshot code to use different detector/matcher combinations.
2. Compare the number of good matches and processing speed for each combination.
3. Fill in the table below:

| Detector | Matcher | Avg. Matches | Avg. FPS |
|----------|---------|--------------|----------|
| SIFT     | BF      |              |          |
| SIFT     | FLANN   |              |          |
| ORB      | BF      |              |          |
| ORB      | FLANN   |              |          |
| AKAZE    | BF      |              |          |
| AKAZE    | FLANN   |              |          |

> **Note:** For FLANN with SIFT, use:
> ```python
> index_params = dict(algorithm=1, trees=5)
> search_params = dict(checks=50)
> flann = cv2.FlannBasedMatcher(index_params, search_params)
> ```

### Task 2: Patterned Sheet Transformation Estimation
**Instructions:**
1. Print the provided pattern sheet.
2. Take two snapshots of the sheet at different positions/orientations.
3. Estimate the transformation (translation and rotation) between the snapshots.
4. Implement real-time transformation estimation:
   - Capture the live webcam feed
   - Detect the pattern in each frame
   - Compute and display the absolute translation and rotation relative to the first snapshot

**Code Snippets for use:**
```python
import cv2
import numpy as np

# Initialize feature detector (ORB recommended for speed)
detector = cv2.ORB_create()

# Pattern detection function
def detect_pattern(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = detector.detectAndCompute(gray, None)
    return kp, des

# Transformation estimation function
def estimate_transform(kp1, des1, kp2, des2):
    # Feature matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Extract points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Find homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # Decompose homography to get rotation and translation
    # Note: This requires camera calibration matrix. For this lab, use:
    # Assume pattern is on a flat surface and use approximate values
    if M is not None:
        dx = M[0, 2]
        dy = M[1, 2]
        rotation = np.arctan2(M[1, 0], M[0, 0])
        return dx, dy, rotation
    return None

```

**Deliverables:**
1. Completed comparison table from Task 1.
2. Code for Task 2 with real-time transformation display.

---

## Conclusion
This lab covered fundamental feature detection and matching techniques. You should now understand the trade-offs between different algorithms and how to implement them for practical applications like transformation estimation.
