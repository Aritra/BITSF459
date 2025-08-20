import cv2
import numpy as np

# Hardcoded image path
IMAGE_PATH = '/home/aritra/CV_codebase/STI/Old_classic/sailboat.bmp'

# Parameters
WINDOW_SIZE = 7  # Size of the window around the clicked point

# Load image
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")

img_disp = img.copy()

def draw_autocorrelation_ellipse(img, x, y, window_size):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Ensure window is within image bounds
    half_w = window_size // 2
    x1, x2 = max(x - half_w, 0), min(x + half_w + 1, gray.shape[1])
    y1, y2 = max(y - half_w, 0), min(y + half_w + 1, gray.shape[0])
    patch = gray[y1:y2, x1:x2].astype(np.float32)

    # Compute gradients
    Ix = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)

    # Compute autocorrelation matrix
    A = np.zeros((2, 2), dtype=np.float32)
    A[0, 0] = np.sum(Ix * Ix)
    A[0, 1] = np.sum(Ix * Iy)
    A[1, 0] = np.sum(Ix * Iy)
    A[1, 1] = np.sum(Iy * Iy)

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(A)
    # Sort eigenvalues and eigenvectors
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Ellipse parameters
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    scale = 0.05
    major_axis = max(1, int(np.sqrt(eigvals[0]) * 2 * scale))
    minor_axis = max(1, int(np.sqrt(eigvals[1]) * 2 * scale))

    # Draw ellipse
    cv2.ellipse(
        img_disp,
        (x, y),
        (major_axis, minor_axis),
        angle,
        0,
        360,
        (0, 0, 255),
        2
    )

def mouse_callback(event, x, y, flags, param):
    global img_disp
    if event == cv2.EVENT_LBUTTONDOWN:
        draw_autocorrelation_ellipse(img, x, y, WINDOW_SIZE)
        cv2.imshow('Image', img_disp)

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)
cv2.imshow('Image', img_disp)
cv2.waitKey(0)
cv2.destroyAllWindows()