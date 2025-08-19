import cv2
"""
Ball Tracker using OpenCV and Open3D
This script captures video from the webcam, detects a colored ball using HSV thresholding and Hough Circle detection,
and visualizes the ball's position and size in a 3D Open3D window. The color range for detection can be adjusted
using OpenCV trackbars.
Functions:
----------
create_ball(radius=1.0, color=[1, 0, 0]):
    Creates a 3D sphere mesh using Open3D with the specified radius and color.
nothing(x):
    Dummy callback function for OpenCV trackbars.
Main Loop:
----------
- Captures frames from the webcam.
- Converts frames to HSV color space.
- Applies color thresholding using trackbar values.
- Performs morphological operations to clean up the mask.
- Detects circles in the mask using cv2.HoughCircles.
- Finds the largest detected circle and updates its position and size.
- Updates the 3D visualization to reflect the ball's movement and scaling.
- Displays the original frame and mask with detected circles.
- Exits when 'q' is pressed.
Dependencies:
-------------
- OpenCV (cv2)
- NumPy
- Open3D
Usage:
------
Run the script. Adjust the trackbars to select the color of the ball to track. The 3D visualization will update in real-time
to reflect the ball's position and size as detected in the video stream.
"""
import numpy as np
import open3d as o3d

# --- Open3D setup ---
def create_cube(size=1.0, color=[1, 0, 0], origin=[0, 0, 0]):
    mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh
def create_ball(radius=1.0, color=[1, 0, 0]):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh

# Create coordinate frame for reference
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5, origin=[0, 0, 0])
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='3D Cube', width=600, height=600)
box = create_cube(size=1.0)
vis.add_geometry(box)
vis.add_geometry(coord_frame)
ctr = vis.get_view_control()

# --- Trackbar callback ---
def nothing(x):
    pass

# --- OpenCV setup ---
cv2.namedWindow('Frame')
cv2.namedWindow('Mask')
cv2.createTrackbar('Lower Hue', 'Frame', 20, 179, nothing)
cv2.createTrackbar('Upper Hue', 'Frame', 30, 179, nothing)

cap = cv2.VideoCapture(0)

# Initial position and scaling
box_pos = np.array([0.0, 0.0, 0.0])
prev_center = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    #apply gaussian blur to reduce noise
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get trackbar positions
    lh = cv2.getTrackbarPos('Lower Hue', 'Frame')
    uh = cv2.getTrackbarPos('Upper Hue', 'Frame')

    # Threshold for color
    lower = np.array([lh, 50, 50])
    upper = np.array([uh, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Morphological operations
    mask = cv2.erode(mask, None, iterations=5)
    mask = cv2.dilate(mask, None, iterations=5)

    # Hough Circle detection
    '''
    Parameters for cv2.HoughCircles:
        - image: Input image (should be grayscale).
        - method: Detection method (cv2.HOUGH_GRADIENT).
        - dp: Inverse ratio of accumulator resolution to image resolution (e.g., dp=1.2 means accumulator has lower resolution).
        - minDist: Minimum distance between detected centers (prevents multiple nearby detections).
        - param1: Higher threshold for Canny edge detector (used internally).
        - param2: Accumulator threshold for circle centers at detection stage (smaller -> more false circles).
        - minRadius: Minimum circle radius to detect.
        - maxRadius: Maximum circle radius to detect (0 means no upper limit).
    '''
    
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=50, param2=25, minRadius=5, maxRadius=0)
    center = None
    radius = None

    # Convert mask to BGR for colored drawing
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Find the circle with the largest radius
        max_circle = max(circles[0, :], key=lambda c: c[2])
        center = (max_circle[0], max_circle[1])
        radius = max_circle[2]
        # Draw the largest circle in red
        cv2.circle(mask_bgr, center, radius, (0, 0, 255), 2)
        cv2.circle(mask_bgr, center, 2, (0, 0, 255), 3)
    else:
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Show frames
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask_bgr)
    
    #get midpoint of the frame
    h, w, _ = frame.shape
    midpointx = w//2
    midpointy = h//2
    

    # --- Open3D update ---
    
    if center is not None and radius is not None:
        if center is not None:
            dx = center[0] - midpointx
            dy = center[1] - midpointy
        else:
            dx, dy = 0, 0

        # Update ball position and size
        xyscalefactor = 0.1
        zscalefactor = -0.1
        new_pos = np.array([dx*xyscalefactor, -dy*xyscalefactor, 0.0])
        new_pos[2] = float(radius*zscalefactor)  # z = radius (scaling factor 1)
        print("dx: "+ str(dx) + " dy: " + str(dy) + " radius: " + str(radius) + " new_pos: " + str(new_pos))
        translation = new_pos - box_pos

        # Remove old ball and add new one
        #vis.clear_geometries()
        #box = create_cube(size=1)
        box.translate(translation)
        #vis.add_geometry(box)
        #vis.add_geometry(coord_frame)
        box_pos = new_pos
        vis.update_geometry(box)
        vis.poll_events()
        vis.update_renderer()
    else:
        vis.poll_events()
        vis.update_renderer()

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
vis.destroy_window()