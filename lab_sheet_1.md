Here’s a **complete GitHub-ready markdown** for your first Computer Vision lab sheet.
I’ve made it structured, with clear sections, fenced code blocks, and task prompts so you can copy-paste it directly into a `.md` file.

---

````markdown
# 🖥️ Computer Vision Lab 1 — Environment Setup & First OpenCV Experiments

## 🎯 Objective
In this lab, you will:
1. Set up a Python development environment using **Miniconda**.
2. Install essential libraries for computer vision and 3D processing.
3. Set up **Visual Studio Code** for Python.
4. Download standard test images for future experiments.
5. Learn basic **OpenCV** operations for reading, displaying, and processing images.
6. Work with live webcam feeds, sliders, and basic image filtering.

---

## 1️⃣ Install Miniconda & Create Python Environment

### **Step 1:** Download & Install Miniconda
- Download Miniconda from: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
- Install with default settings.

### **Step 2:** Create a new environment
```bash
conda create --name cvlab python=3.10
conda activate cvlab
````

### **Step 3:** Install required packages

```bash
pip install opencv-python opencv-contrib-python matplotlib open3d typing_extensions
```

> You can add more packages as needed in future labs.

---

## 2️⃣ Setting up VS Code for Python

1. Download VS Code: [https://code.visualstudio.com/](https://code.visualstudio.com/)
2. Install the **Python extension** from the Extensions Marketplace.
3. Select the Python interpreter for your `cvlab` environment:

   * Press `Ctrl+Shift+P`
   * Search for **Python: Select Interpreter**
   * Choose `cvlab` environment.

---

## 3️⃣ Download Standard Test Images

### Clone GitHub Repository

```bash
git clone https://github.com/mohammadimtiazz/standard-test-images-for-Image-Processing.git
```

### Download Dataset from Kaggle

* Go to: [https://www.kaggle.com/datasets/saeedehkamjoo/standard-test-images?resource=download](https://www.kaggle.com/datasets/saeedehkamjoo/standard-test-images?resource=download)
* Download and extract the dataset into your working directory.

---

## 4️⃣ Basic Image Operations in OpenCV

### Reading & Displaying Images

```python
import cv2

# Read an image
img = cv2.imread("lena.png")  # path to your image

# Display in a window
cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### Checking Image Properties

```python
print("Shape (Height, Width, Channels):", img.shape)
print("Resolution:", img.shape[1], "x", img.shape[0])
print("Number of Channels:", img.shape[2])
```

---

### Convert to Grayscale

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### Writing Images to File

```python
cv2.imwrite("output.png", gray)
```

---

### Pixel Value Manipulation

#### Thresholding for B/W

```python
_, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
cv2.imshow("Black & White", bw)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### Color Inversion

```python
inverted = 255 - img
cv2.imshow("Inverted Image", inverted)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 5️⃣ Accessing Webcam & Adjusting Brightness/Contrast

### Webcam Stream

```python
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Webcam", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
```

---

### Webcam with Brightness/Contrast Sliders

```python
def adjust_contrast_brightness(image, contrast, brightness):
    return cv2.convertScaleAbs(image, alpha=contrast/50, beta=brightness-50)

cap = cv2.VideoCapture(0)

cv2.namedWindow("Adjustments")
cv2.createTrackbar("Contrast", "Adjustments", 50, 100, lambda x: None)
cv2.createTrackbar("Brightness", "Adjustments", 50, 100, lambda x: None)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    contrast = cv2.getTrackbarPos("Contrast", "Adjustments")
    brightness = cv2.getTrackbarPos("Brightness", "Adjustments")

    adjusted = adjust_contrast_brightness(frame, contrast, brightness)
    
    cv2.imshow("Adjustments", adjusted)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 6️⃣ First Task (Non-Evaluative) — Face Hue Detection

🎯 **Goal:** Convert webcam feed to HSV and use hue-based filtering to detect face region.

---

## Aditional task

🎯 **Goal:** Read the webcam feed and plot live histogram

The following code snippet is for your reference to use:

Intensity histogram for single-channel images.

Separate B, G, R histograms for 3-channel images.

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("lena.png")
channels = len(img.shape)

if channels == 2 or img.shape[2] == 1:
    # Grayscale
    gray = img if channels == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.hist(gray.ravel(), 256, [0, 256], color='black')
    plt.title("Intensity Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.show()
else:
    # BGR Histograms
    color = ('b', 'g', 'r')
    plt.figure(figsize=(10,5))
    for i, col in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.title("BGR Histograms")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.show()
```
---
