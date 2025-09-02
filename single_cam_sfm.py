import cv2
import numpy as np
import os


def save_ply(filename, points, colors):
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for p, c in zip(points, colors):
            f.write(f'{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n')

def get_sift_keypoints_and_descriptors(img):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des

def match_features(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good

def get_matched_points(kp1, kp2, matches):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2

def triangulate_points(P1, P2, pts1, pts2):
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = pts4d[:3] / pts4d[3]
    return pts3d.T

def main():
    cam_K_path = os.path.join(os.path.dirname(__file__), 'cam_K.npy')
    cam_K = np.load(cam_K_path)
    snapshots = []
    colors = []

    cap = cv2.VideoCapture(0)
    print("Press SPACE to take snapshot, 'd' to process and exit, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Webcam', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            snapshots.append(frame.copy())
            print(f"Snapshot taken: {len(snapshots)}")
        elif key == ord('d'):
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
    cap.release()
    cv2.destroyAllWindows()

    if len(snapshots) < 2:
        print("Need at least 2 snapshots for SfM.")
        return

    all_points = []
    all_colors = []

    for i in range(len(snapshots)-1):
        img1 = cv2.cvtColor(snapshots[i], cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(snapshots[i+1], cv2.COLOR_BGR2GRAY)
        kp1, des1 = get_sift_keypoints_and_descriptors(img1)
        kp2, des2 = get_sift_keypoints_and_descriptors(img2)
        matches = match_features(des1, des2)
        if len(matches) < 8:
            print("Not enough matches between images.")
            continue
        pts1, pts2 = get_matched_points(kp1, kp2, matches)
        # Ensure correct shape and type
        pts1 = np.asarray(pts1, dtype=np.float32)
        pts2 = np.asarray(pts2, dtype=np.float32)
        E, mask = cv2.findEssentialMat(pts1, pts2, cam_K, method=cv2.RANSAC, prob=0.999, threshold=3.0)
        if E is None:
            print("Essential matrix could not be computed.")
            continue
        pts1_inliers_pre_pose = pts1[mask.ravel() == 1]
        pts2_inliers_pre_pose = pts2[mask.ravel() == 1]
        print(f"Number of inliers after findEssentialMat: {pts1_inliers_pre_pose.shape[0]}")

        # Use the filtered inliers for recoverPose
        _, R, t, mask_pose = cv2.recoverPose(E, pts1_inliers_pre_pose, pts2_inliers_pre_pose, cam_K)
        pts1_inliers = pts1_inliers_pre_pose[mask_pose.ravel() == 1]
        pts2_inliers = pts2_inliers_pre_pose[mask_pose.ravel() == 1]
        print(f"Number of inliers after recoverPose: {pts1_inliers.shape[0]}")
        
        # pts1_inliers = pts1[mask_pose.ravel() == 1]  #finding inliers
        # pts2_inliers = pts2[mask_pose.ravel() == 1]  #finding inliers
        # pts1_inliers = pts1
        # pts2_inliers = pts2
        if pts1_inliers.shape[0] < 8:
            print("Not enough inlier points after pose recovery.")
            continue
        P1 = cam_K @ np.hstack((np.eye(3), np.zeros((3,1))))
        P2 = cam_K @ np.hstack((R, t))
        # Triangulate points: input must be 2xN
        points1_homog = np.vstack((pts1_inliers, np.ones((1, pts1_inliers.shape[1]))))
        points2_homog = np.vstack((pts2_inliers, np.ones((1, pts2_inliers.shape[1]))))
        
        points1_homog = pts1_inliers.T
        points2_homog = pts2_inliers.T
        
        print('P1 shape:', P1.shape)
        print('P2 shape:', P2.shape)
        print('points1_homog shape:', points1_homog.shape)
        print('points2_homog shape:', points2_homog.shape)
        
        points_3d_homog = cv2.triangulatePoints(P1, P2, points1_homog, points2_homog)

        # Convert 3D homogeneous points to Cartesian coordinates
        pts3d = points_3d_homog[:3] / points_3d_homog[3]
        
        #pts3d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        #pts3d = pts3d_hom[:3] / pts3d_hom[3]
        pts3d = pts3d.T  # shape (N, 3)
        # Get colors from first image
        h, w = snapshots[i].shape[:2]
        for pt in pts1:
            x, y = int(round(pt[0])), int(round(pt[1]))
            if 0 <= x < w and 0 <= y < h:
                color = snapshots[i][y, x]
                all_colors.append(color)
        all_points.extend(pts3d)
       

    if len(all_points) == 0:
        print("No 3D points reconstructed.")
        return

    all_points = np.array(all_points)
    all_colors = np.array(all_colors)
    save_ply('3d_pc.ply', all_points, all_colors)
    print(f"Saved 3D point cloud to 3d_pc.ply")

if __name__ == '__main__':
    main()