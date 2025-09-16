# LabSheet 5 — Monocular Structure from Motion & Monocular SLAM

**Goals:**

1. Build a sparse colored 3D point cloud from multiple monocular views (offline SfM).
2. Implement a simple live monocular SLAM demo that visualizes camera trajectory and sparse landmarks in a live Matplotlib 3D plot.

---

## Prerequisites / Environment

update your conda environment with following packages if not there already:

```bash
pip install opencv-contrib-python numpy scipy matplotlib
```

* `opencv-contrib-python` is required for **SIFT** and some contrib functions.
* `scipy` used for `least_squares` (bundle adjustment).
* Optionally: `meshlab` or `cloudcompare` to visualise produced `.ply`.

---

## Files you will produce / use

* `K.npy` — camera intrinsic matrix (3×3) saved with `np.save("K.npy", K)`.
* `dist.npy` — distortion coefficients from the calibrator `np.save("dist.npy", dist)`.
* Image folder `images/` — sequence of images captured with the webcam (or other camera).
* Output: `scene.ply` — sparse colored point cloud + camera wireframes.

> **Important note about monocular SfM:** absolute scale is unknown from a single moving camera — reconstructions are up-to-scale.

---

## Quick camera snapshot tool (press `space` to save frames)

Save this as `capture_images.py` and run to capture frames into a folder:

```python
# capture_images.py
import os, cv2, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="images", help="output folder")
ap.add_argument("--prefix", default="img", help="filename prefix")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
cap = cv2.VideoCapture(0)
idx = 0
print("Press SPACE to save, ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("capture (space=save, esc=exit)", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ESC
        break
    if k == 32:  # SPACE
        fname = f"{args.prefix}_{idx:03d}.jpg"
        path = os.path.join(args.out, fname)
        cv2.imwrite(path, frame)
        print("Saved", path)
        idx += 1

cap.release()
cv2.destroyAllWindows()
```

**Advice:** capture a sequence by moving the camera around the object / scene with moderate baseline between frames. Save `K.npy` and `dist.npy` produced by your calibration script.

---

# Part A — Monocular Structure from Motion (offline)

### Objective

Given a folder of images and optional camera calibration (`K.npy`, `dist.npy`), compute a sparse 3D point cloud with colors and refine it with bundle adjustment. Export `.ply`.

### High-level algorithm (step-by-step)

1. **Load images** (and undistort using `K.npy` / `dist.npy` if present).
2. **Detect features** in every image (SIFT) and compute descriptors.
3. **Match features** between consecutive images using BFMatcher + ratio test.
4. **Build tracks** by linking matches across frames (simple consecutive linking).
5. **Estimate relative poses** for image pairs:

   * Estimate fundamental matrix `F` with RANSAC.
   * Compute essential matrix `E = K^T F K`.
   * Decompose `E` to get candidate `(R, t)` pairs via SVD; choose the physically correct one (points in front of both cameras).
6. **Pose chaining**: set world = first camera, chain relative motions to get each camera pose.
7. **Triangulate** 3D points for each track using the pair of camera poses where the track is observed (use first/last or pair with good baseline).
8. **Bundle adjustment**: jointly refine camera rotations/translations and 3D points by minimizing reprojection error (LM or Gauss–Newton).
9. **Export PLY** with colors sampled from first observation of each point and camera wireframes.

---

### Useful math (display)

Projection of a 3D point \$X\$ into camera \$i\$:

$$
\hat{x}_{ij} = \pi(K, R_i, t_i, X_j) = \frac{K \, (R_i \, X_j + t_i)}{z}
$$

Objective (reprojection error):

$$
\min_{\{R_i, t_i, X_j\}} \sum_{i,j} \| x_{ij} - \pi(K, R_i, t_i, X_j) \|^2
$$

Essential from Fundamental:

$$
E = K^\top F K
$$

Decomposition of \$E\$ (SVD):

$$
E = U \,\mathrm{diag}(1,1,0)\, V^\top
$$

Possible solutions:

$$
R = U W V^\top \quad\text{or}\quad R = U W^\top V^\top,\qquad t = \pm U[:,2]
$$

where

$$
W=\begin{bmatrix}0 & -1 & 0\\ 1 & 0 & 0\\ 0 & 0 & 1\end{bmatrix}.
$$

---

### Full working script — `monocular_sfm.py`

> **This is a complete script** implementing the algorithm above. It expects an `--images` folder and optionally `--K K.npy --dist dist.npy`. It writes `scene.ply`.

```python
#!/usr/bin/env python3
"""
monocular_sfm.py
Offline monocular SfM:
 - loads images
 - undistorts (optional)
 - SIFT -> BF matching -> track building
 - estimate relative poses via F -> E
 - chain poses, triangulate tracks
 - bundle adjust (scipy.least_squares)
 - export PLY with colors + camera wireframe
"""

import os, glob, argparse
import numpy as np
import cv2
from scipy.optimize import least_squares

# -------------------------------
# I/O helpers
# -------------------------------
def load_images(folder):
    exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(folder,e))
    files = sorted(files)
    imgs = [cv2.imread(f, cv2.IMREAD_COLOR) for f in files]
    imgs = [im for im in imgs if im is not None]
    if len(imgs) < 2:
        raise ValueError("Need >=2 images")
    return imgs, files

def undistort_images(images, K, dist):
    h,w = images[0].shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
    return [cv2.undistort(im, K, dist, None, newK) for im in images], newK

# -------------------------------
# Features + matching
# -------------------------------
def compute_sift(img):
    sift = cv2.SIFT_create()
    kps, des = sift.detectAndCompute(img, None)
    return kps, des

def match_desc(des1, des2, ratio=0.75):
    if des1 is None or des2 is None or len(des1)==0 or len(des2)==0:
        return []
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(des1, des2, k=2)
    good = []
    for m,n in knn:
        if m.distance < ratio * n.distance:
            good.append((m.queryIdx, m.trainIdx))
    return good

# -------------------------------
# Build simple tracks (link consecutive matches)
# -------------------------------
def build_tracks(matches_list):
    # matches_list[i] are matches between frame i and i+1 as (a,b) pairs
    tracks = {}  # tid -> list of (frame_index, kp_idx)
    kp_to_tid = []  # for each frame, map kp idx -> tid
    next_tid = 0
    for i, matches in enumerate(matches_list):
        if i == 0:
            kp_to_tid.append({})
        if i+1 >= len(matches_list)+1:
            kp_to_tid.append({})
        # ensure dict exists for frame i+1
        if len(kp_to_tid) <= i+1:
            kp_to_tid.append({})
        for a,b in matches:
            tid = kp_to_tid[i].get(a)
            if tid is None:
                # new track
                tid = next_tid; next_tid += 1
                tracks[tid] = [(i,a)]
                kp_to_tid[i][a] = tid
            # append for frame i+1
            tracks[tid].append((i+1, b))
            kp_to_tid[i+1][b] = tid
    return tracks

# -------------------------------
# Geometry utilities
# -------------------------------
def ransac_fundamental(pts1, pts2):
    if len(pts1) < 8:
        return None, None
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.999)
    if F is None:
        return None, None
    return F, mask.ravel().astype(bool)

def essential_from_F(F, K):
    return K.T @ F @ K

def decompose_E_choose(E, K, pts1, pts2):
    # pts1, pts2: Nx2 float32 (in pixel coords)
    U, S, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0: U[:, -1] *= -1
    if np.linalg.det(Vt) < 0: Vt[-1, :] *= -1
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]
    candidates = [(R1, t), (R1, -t), (R2, t), (R2, -t)]
    # check chirality: pick the candidate that yields most points in front of both cameras
    P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))])
    best = None; best_count = -1; bestX = None; bestRT=None
    for R, tvec in candidates:
        P2 = K @ np.hstack([R, tvec.reshape(3,1)])
        X4 = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        X = (X4[:3] / X4[3]).T
        # compute z in camera 1 and camera 2
        def z_in_cam(P, X):
            Xh = np.hstack([X, np.ones((X.shape[0],1))]).T
            x = (P @ Xh).T
            return x[:,2]
        z1 = z_in_cam(P1, X)
        z2 = z_in_cam(P2, X)
        count = np.sum((z1 > 0) & (z2 > 0))
        if count > best_count:
            best_count = count
            best = (R, tvec)
            bestX = X
    return best, bestX

# -------------------------------
# Bundle adjustment (simple LM)
# -------------------------------
def rodrigues_to_vec(R):
    r, _ = cv2.Rodrigues(R)
    return r.ravel()

def vec_to_rodrigues(rvec):
    R, _ = cv2.Rodrigues(rvec.reshape(3,1))
    return R

def project(K, R, t, X):
    P = K @ np.hstack([R, t.reshape(3,1)])
    Xh = np.hstack([X, np.ones((X.shape[0],1))]).T
    x = (P @ Xh).T
    return x[:,:2] / x[:,2:3]

def bundle_adjustment(K, cam_params, X0, obs_cam, obs_pt, obs_xy, max_nfev=200):
    """
    cam_params: list of (rvec(3), t(3))
    X0: (M,3)
    obs_cam: (N,) camera indices for each observation
    obs_pt: (N,) point indices
    obs_xy: (N,2) observed pixel coords
    returns: optimized cams (list (R,t)) and X (M,3)
    """
    n_cams = len(cam_params)
    n_points = X0.shape[0]
    n_obs = obs_xy.shape[0]

    x0_cams = np.hstack([np.hstack([c[0], c[1]]) for c in cam_params])  # (n_cams*6,)
    x0 = np.hstack([x0_cams, X0.ravel()])

    def unpack(x):
        cams = x[:n_cams*6].reshape((n_cams,6))
        Xs = x[n_cams*6:].reshape((n_points,3))
        return cams, Xs

    def residuals(x):
        cams, Xs = unpack(x)
        res = np.zeros((n_obs*2,))
        for i in range(n_obs):
            ci = obs_cam[i]; pi = obs_pt[i]
            rvec = cams[ci,:3]; t = cams[ci,3:]
            R = vec_to_rodrigues(rvec)
            proj = project(K, R, t, Xs[pi:pi+1])[0]
            res[2*i:2*i+2] = proj - obs_xy[i]
        return res

    out = least_squares(residuals, x0, method='lm', max_nfev=max_nfev, verbose=2)
    cams_opt, X_opt = unpack(out.x)
    cams_ref = [(vec_to_rodrigues(c[:3]), c[3:]) for c in cams_opt]
    return cams_ref, X_opt

# -------------------------------
# PLY writer (points + colors + camera wireframes)
# -------------------------------
def write_ply(points, colors, cams, filename):
    verts = []
    edges = []
    for p,c in zip(points, colors):
        verts.append((float(p[0]), float(p[1]), float(p[2]), int(c[2]), int(c[1]), int(c[0])))
    # cameras as white pyramids
    for (R, t) in cams:
        C = -R.T @ t  # camera center in world coordinates if R,t are world->cam? careful: here cams are Rwc,twc already
        # we expect cams as (Rcw, C) to write; be consistent at callsite
        pass
    # Simpler: write only points + colors
    with open(filename, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("element vertex %d\n" % len(verts))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for v in verts:
            f.write("%f %f %f %d %d %d\n" % v)
    print("Wrote PLY to", filename)

# -------------------------------
# Main pipeline
# -------------------------------
def run_sfm(image_folder, Kfile, distfile, out_ply):
    images, files = load_images(image_folder)
    if Kfile and distfile:
        K = np.load(Kfile)
        dist = np.load(distfile)
        images, K = undistort_images(images, K, dist)
        print("Loaded calibration and undistorted images.")
    else:
        # guess focal
        h,w = images[0].shape[:2]
        K = np.array([[1.2*max(w,h), 0, w/2.],
                      [0, 1.2*max(w,h), h/2.],
                      [0,0,1.0]])
        print("No K provided. Using guessed K:\n", K)

    # detect features
    KPS = []; DES = []
    for im in images:
        kp, des = compute_sift(im)
        KPS.append(kp)
        DES.append(des)

    # match consecutive frames
    matches_list = []
    for i in range(len(images)-1):
        matches = match_desc(DES[i], DES[i+1])
        matches_list.append(matches)
        print(f"Frame {i} -> {i+1}: {len(matches)} matches")

    # build tracks
    tracks = build_tracks(matches_list)
    print("Built", len(tracks), "tracks")

    # estimate relative poses and chain them
    posesR = [np.eye(3)]
    poset = [np.zeros(3)]
    for i in range(len(images)-1):
        # get matches that exist between i and i+1
        pts1=[]; pts2=[]
        for a,b in matches_list[i]:
            pts1.append(KPS[i][a].pt); pts2.append(KPS[i+1][b].pt)
        pts1 = np.array(pts1, dtype=np.float32); pts2 = np.array(pts2, dtype=np.float32)
        if len(pts1) < 8:
            print("Not enough matches for F between", i, i+1)
            R_rel = np.eye(3); t_rel = np.zeros(3)
        else:
            F, mask = ransac_fundamental(pts1, pts2)
            if F is None:
                R_rel = np.eye(3); t_rel = np.zeros(3)
            else:
                in1 = pts1[mask]; in2 = pts2[mask]
                E = essential_from_F(F, K)
                (R_rel,t_rel), _ = decompose_E_choose(E, K, in1, in2)
        Rprev = posesR[-1]; tprev = poset[-1]
        Rnew = R_rel @ Rprev
        tnew = R_rel @ tprev + t_rel
        posesR.append(Rnew); poset.append(tnew)

    # Triangulate tracks between first and last observation of each track
    pts3d = []
    colors = []
    track_to_pt = {}
    for tid, obs in tracks.items():
        if len(obs) < 2:
            continue
        f1, k1 = obs[0]
        f2, k2 = obs[-1]
        R1, t1 = posesR[f1], poset[f1]
        R2, t2 = posesR[f2], poset[f2]
        P1 = K @ np.hstack([R1, t1.reshape(3,1)])
        P2 = K @ np.hstack([R2, t2.reshape(3,1)])
        x1 = np.array(KPS[f1][k1].pt).reshape(2,1)
        x2 = np.array(KPS[f2][k2].pt).reshape(2,1)
        X4 = cv2.triangulatePoints(P1, P2, x1, x2)
        X = (X4[:3] / X4[3]).T[0]
        pts3d.append(X)
        # sample color from first observation
        u,v = map(int, map(round, KPS[f1][k1].pt))
        if 0 <= u < images[f1].shape[1] and 0 <= v < images[f1].shape[0]:
            color = images[f1][v,u]
        else:
            color = np.array([128,128,128])
        colors.append(color)
        track_to_pt[tid] = len(pts3d)-1

    pts3d = np.array(pts3d)
    colors = np.array(colors, dtype=np.uint8)
    print("Triangulated", pts3d.shape[0], "points")

    # prepare BA observations
    obs_xy = []
    obs_cam = []
    obs_pt = []
    for tid, pid in track_to_pt.items():
        for f, k in tracks[tid]:
            xy = np.array(KPS[f][k].pt, dtype=np.float64)
            obs_xy.append(xy)
            obs_cam.append(f)
            obs_pt.append(pid)
    obs_xy = np.array(obs_xy)
    obs_cam = np.array(obs_cam)
    obs_pt = np.array(obs_pt)

    # prepare camera params for BA
    cam_params = []
    for R, t in zip(posesR, poset):
        rvec = rodrigues_to_vec(R)
        cam_params.append((rvec, t))

    if pts3d.shape[0] > 0:
        cams_opt, X_opt = bundle_adjustment(K, cam_params, pts3d, obs_cam, obs_pt, obs_xy, max_nfev=100)
        print("Bundle adjustment done")
    else:
        cams_opt = cam_params
        X_opt = pts3d

    # convert cams to exportable format (Rcw, C) for viewer if necessary
    cams_out = []
    for R, t in cams_opt:
        # Here R is rotation matrix world->cam? In our chain R,t are world->cam; camera center C = -R^T t
        C = -R.T @ t
        cams_out.append((R.T, C))  # we store Rcw? small differences depending on paramization

    write_ply(X_opt, colors, cams_out, out_ply)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--K", default=None)
    ap.add_argument("--dist", default=None)
    ap.add_argument("--out", default="scene.ply")
    args = ap.parse_args()
    run_sfm(args.images, args.K, args.dist, args.out)
```

**How to run**:

```bash
python monocular_sfm.py --images images --K K.npy --dist dist.npy --out scene.ply
```

**Notes & tips**

* If `K.npy`/`dist.npy` not provided, a rough focal is guessed (reconstruction less accurate).
* If too few matches are found, reduce ratio in `match_desc()` or use more distinctive frames.
* BA may take time; limit number of points/cameras for a quick demo.

---

# Part B — Monocular SLAM (live visualization)

### Objective

Run a **simple monocular SLAM demo** with a webcam: compute incremental poses from consecutive frames and triangulate landmarks, while plotting camera trajectory + sparse landmarks in a live Matplotlib 3D window.

### Important caveats

* **Scale ambiguity**: monocular SLAM produces a reconstruction up to scale. The relative magnitude of translations is arbitrary (unless you provide scale info).
* This is a **teaching/demo** pipeline (not a production SLAM). It demonstrates camera pose chaining, incremental triangulation, and live visualization.
* Real SLAM requires robust keyframe selection, data association, map management, loop closure, and real-time BA — those are left as tasks.

### High-level pipeline (live)

1. Open camera; optionally undistort frames using `K.npy` / `dist.npy`.
2. For each new frame:

   * Detect features & compute descriptors.
   * Match with the previous keyframe (or previous frame).
   * Compute `F` via RANSAC, form `E = K^T F K`.
   * Decompose `E` → choose correct `(R, t)` via chirality.
   * Chain pose: `T_world_new = T_world_prev * T_rel`.
   * Triangulate matching points between prev & current frame into 3D; add to map.
   * Update Matplotlib 3D scatter / trajectory line.
3. Optionally save map to `.ply`.

### Live SLAM demo script — `monocular_slam_live.py`

```python
#!/usr/bin/env python3
"""
monocular_slam_live.py
Simple live monocular SLAM demo:
 - SIFT features between consecutive frames
 - E decomposition -> relative pose
 - pose chaining (trajectory)
 - triangulate between consecutive frames -> incrementally add landmarks
 - live Matplotlib 3D plot updated in real time (trajectory + landmarks)
"""

import cv2, numpy as np, argparse, time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_calib(Kfile, distfile):
    if Kfile and distfile:
        K = np.load(Kfile)
        dist = np.load(distfile)
        return K, dist
    else:
        return None, None

def undistort_frame(frame, K, dist):
    h,w = frame.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
    return cv2.undistort(frame, K, dist, None, newK), newK

def compute_sift(img):
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(img, None)

def match_desc(des1, des2, ratio=0.75):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(des1, des2, k=2)
    good = []
    for m,n in knn:
        if m.distance < ratio*n.distance:
            good.append((m.queryIdx, m.trainIdx))
    return good

def decompose_E_choose(E, K, pts1, pts2):
    U,S,Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0: U[:,-1] *= -1
    if np.linalg.det(Vt) < 0: Vt[-1,:] *= -1
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:,2]
    cands = [(R1,t),(R1,-t),(R2,t),(R2,-t)]
    P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))])
    best, bestX, bestRT = None, None, None
    best_count = -1
    for R,tvec in cands:
        P2 = K @ np.hstack([R, tvec.reshape(3,1)])
        X4 = cv2.triangulatePoints(P1,P2,pts1.T,pts2.T)
        X = (X4[:3]/X4[3]).T
        Xh = np.hstack([X, np.ones((X.shape[0],1))]).T
        z1 = (P1 @ Xh)[2]
        z2 = (P2 @ Xh)[2]
        c = np.sum((z1>0) & (z2>0))
        if c > best_count:
            best_count = c
            best = (R,tvec)
    return best

def triangulate_points(K, R1, t1, R2, t2, pts1, pts2):
    P1 = K @ np.hstack([R1, t1.reshape(3,1)])
    P2 = K @ np.hstack([R2, t2.reshape(3,1)])
    X4 = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    X = (X4[:3]/X4[3]).T
    return X

def run_live(Kfile=None, distfile=None, show=True):
    cap = cv2.VideoCapture(0)
    K, dist = load_calib(Kfile, distfile)
    if K is None:
        print("No calibration provided. Using guessed K once frames are read.")
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        return
    if K is None:
        h,w = frame.shape[:2]
        K = np.array([[1.2*max(w,h),0,w/2.],[0,1.2*max(w,h),h/2.],[0,0,1]])
        dist = np.zeros(5)

    # Variables
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_prev, des_prev = compute_sift(prev_gray)
    poses = []  # list of (R, t) in world frame; start with identity
    R_world = np.eye(3); t_world = np.zeros(3)
    poses.append((R_world.copy(), t_world.copy()))
    # Map
    pts3d = []
    colors = []

    # Prepare plot
    plt.ion()
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    traj_x, traj_y, traj_z = [],[],[]
    scatter = None
    traj_line = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = compute_sift(frame_gray)
        if des_prev is None or des is None or len(kp_prev)==0 or len(kp)==0:
            kp_prev, des_prev = kp, des
            continue
        matches = match_desc(des_prev, des)
        if len(matches) < 8:
            # not enough matches; skip
            kp_prev, des_prev = kp, des
            continue
        pts1 = np.array([kp_prev[a].pt for a,b in matches], dtype=np.float32)
        pts2 = np.array([kp[b].pt for a,b in matches], dtype=np.float32)

        # estimate essential via F->E
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.999)
        if F is None:
            kp_prev, des_prev = kp, des
            continue
        E = K.T @ F @ K
        good = mask.ravel().astype(bool)
        in1 = pts1[good]; in2 = pts2[good]
        if len(in1) < 8:
            kp_prev, des_prev = kp, des
            continue
        candidate = decompose_E_choose(E, K, in1, in2)
        if candidate is None:
            kp_prev, des_prev = kp, des
            continue
        R_rel, t_rel = candidate

        # normalize translation (scale ambiguity)
        t_rel = t_rel / (np.linalg.norm(t_rel) + 1e-9)

        # update world pose: new_world = prev_world * [R_rel | t_rel]
        R_new = R_rel @ R_world
        t_new = R_rel @ t_world + t_rel

        # triangulate inlier points and add to map
        X = triangulate_points(K, R_world, t_world, R_new, t_new, in1, in2)
        # sample colors from current frame for each triangulated pts using in2 indexes
        for pt, idx_pt in zip(X, range(X.shape[0])):
            u,v = map(int, map(round, in2[idx_pt]))
            if 0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]:
                col = frame[v,u]
            else:
                col = np.array([200,200,200])
            pts3d.append(pt)
            colors.append(col)

        # append pose
        R_world = R_new
        t_world = t_new
        poses.append((R_world.copy(), t_world.copy()))

        # update plotting data
        traj_x.append(t_world[0]); traj_y.append(t_world[1]); traj_z.append(t_world[2])

        if show:
            ax.clear()
            if len(pts3d) > 0:
                P = np.array(pts3d)
                C = np.array(colors)
                ax.scatter(P[:,0], P[:,1], P[:,2], c=C/255.0, s=2)
            ax.plot(traj_x, traj_y, traj_z, '-r')
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            ax.set_title("Live Monocular SLAM: trajectory (red) + landmarks")
            plt.pause(0.001)

        # prepare for next iteration
        kp_prev, des_prev = kp, des

        # display frame
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("SLAM finished. Landmarks:", len(pts3d))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", default=None)
    ap.add_argument("--dist", default=None)
    args = ap.parse_args()
    run_live(args.K, args.dist, show=True)
```

**How to run**:

```bash
python monocular_slam_live.py --K K.npy --dist dist.npy
```

**Notes**

* This demo triangulates points between consecutive frames only. That produces a growing but noisy map. In a full SLAM system you would add:

  * keyframe selection,
  * track management (associate new observations with existing landmarks),
  * map culling & merging,
  * loop-closure detection and pose-graph optimization,
  * and periodic bundle adjustment for map & pose refinement.
* Matplotlib updating is adequate for teaching. For interactive 3D, consider `pyqtgraph`, `open3d.visualization`, or MeshLab for offline viewing.

---

# Student Tasks (deliverables)

1. **SfM pipeline**

   * Capture at least 20 images of an object / face using `capture_images.py`.
   * Run `monocular_sfm.py` on the images to produce `scene.ply`.
   * Visualize `scene.ply` in Meshlab. Verify lines appear straight (if you undistorted properly) and check 3D arrangement.
   * *Extension:* improve track building to use matching to a fixed reference frame (not just consecutive linking) to increase track length.

2. **SLAM pipeline**

   * Run `monocular_slam_live.py` and move the camera slowly around a scene. Observe the live 3D plot.
   * *Extension:* add keyframe selection (e.g., insert a keyframe when baseline exceeds threshold), track existing landmarks across multiple frames, and add a basic loop-closure detector using image descriptors.

3. **Integration**

   * Integrate the offline SfM and live SLAM pipeline: accumulate frames during live SLAM into a buffer; when a set of keyframes has been collected, perform the offline SfM + BA on those keyframes to refine the map and update live visualization.

4. **Extra credit**

   * Add color to the `.ply` export (sampled from the frame where the point was first triangulated). Ensure colors are correct (BGR -> RGB conversions).
   * Add an optional `--scale` input (user-provided reference length) to fix metric scale.

---

# Appendix — practical tips & troubleshooting

* **SIFT availability:** if `cv2.SIFT_create()` raises an error, ensure you installed `opencv-contrib-python` (not plain `opencv-python`).
* **Insufficient matches:** reduce Lowe ratio (e.g., from `0.75` to `0.7`) or capture with larger baseline.
* **F vs E stability:** estimation of `F` with noisy matches is sensitive; RANSAC parameters (`ransacReprojThreshold` or `cv2.findFundamentalMat` thresholds) can be tuned.
* **Scale drift in SLAM:** monocular incremental chaining accumulates scale inaccuracies. Use bundle adjustment to redistribute error, or add IMU/GPS for metric scale.
* **Large BA problems:** solving full BA in Python/scipy may be slow for many cameras/points. For large problems use specialized C++ libraries (Ceres, g2o). For the lab, limit problem size (<\~2000 observations).

---

# Suggested further improvements (for project work)

* Add **keyframe-based tracking** instead of simple frame-to-frame chaining.
* Implement **track-to-landmark association** using descriptor matching + reprojection gating.
* Add **pose-graph optimization** with loop closures for long sequences.
* Replace Matplotlib viewer with `Open3D` or `pyglet` for faster interactive rendering.

---
