

# Lab Sheet 7 — Classical ML Methods for Image Classification

It covers **classical ML for image classification**: Bag-of-Visual-Words (BoW) with SIFT, HOG, and classifiers (Decision Tree, SVM). It includes dataset instructions (MNIST CSV → images, CIFAR-10 loader), data format expectations, and **complete example scripts** (training + testing) for two pipelines:

* **A. MNIST → Decision Tree** (CSV → images → DecisionTreeClassifier)
* **B. CIFAR-10 → HOG features → SVM** (train.py / test.py)
* **C. How to build a BoW codebook with SIFT (example script and usage)**

Make sure your environment contains at least:

```bash
python >= 3.8
numpy, scipy, scikit-learn, opencv-contrib-python, matplotlib
pip install numpy scipy scikit-learn opencv-contrib-python matplotlib
```

## 1. Overview & Motivation

* Classical image classification relies on (1) good **hand-crafted features** and (2) supervised classifiers.
* Two important classical features:

  * **Bag of Visual Words (BoW)**: uses local descriptors (e.g., SIFT) → cluster descriptors to form a visual vocabulary (codebook) → represent each image as histogram of visual words. Good for object recognition / retrieval.
  * **Histogram of Oriented Gradients (HOG)**: dense gradient histogram per cell → robust for shape/texture (used in pedestrian detection, etc.).
* Common classifiers: **Support Vector Machine (SVM)**, **Decision Tree**, **Random Forest**, **k-NN** (we use SVM & Decision Tree in this lab).
* Libraries: use **OpenCV** for images and feature extraction, **scikit-learn** (sklearn) for clustering, encoders, and classifiers.

---

## 2. Data: CIFAR-10 and MNIST

### CIFAR-10

* Source: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
* CIFAR-10 contains 60k color images of size 32×32 in 10 classes.
* We provide a small loader below that reads the original binary batches (or you can use `keras.datasets.cifar10` if available).

### MNIST (CSV)

* Kaggle mirror: [https://www.kaggle.com/datasets/oddrationale/mnist-in-csv](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
* Contains a CSV where each row is `label,pixel0,pixel1,...,pixel783`.
* Students must convert CSV to image files to be used by OpenCV / feature extractors and to create the dataset folder structure.

---

## 3. Expected Data Format (for training scripts)

The training / testing scripts expect the following folder layout for image datasets (both CIFAR preprocessed into image files and MNIST converted):

```
data/
  train/
    0/      # class 0
      img_0001.png
      img_0002.png
      ...
    1/
    ...
    9/
  test/
    0/
    ...
```

* Each class folder contains PNG/JPG images. This is a standard and simple format to implement a dataloader.

---

## 4. Utility: Convert MNIST CSV → images

Save this script as `mnist_csv_to_images.py`. It converts the CSV to `data/train/` and `data/test/` directories (or a single folder).

```python
# mnist_csv_to_images.py
import os, csv, argparse
import numpy as np
import cv2

def csv_to_images(csv_file, out_dir, limit=None):
    os.makedirs(out_dir, exist_ok=True)
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # if CSV has header, else comment out
        count = 0
        for row in reader:
            if limit is not None and count >= limit:
                break
            label = row[0]
            pixels = np.array(row[1:], dtype=np.uint8)
            img = pixels.reshape(28,28)
            class_dir = os.path.join(out_dir, label)
            os.makedirs(class_dir, exist_ok=True)
            fname = os.path.join(class_dir, f"{label}_{count:06d}.png")
            cv2.imwrite(fname, img)
            count += 1
    print("Saved", count, "images to", out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="data/mnist_images/train")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    csv_to_images(args.csv, args.out, args.limit)
```

Usage example:

```bash
python mnist_csv_to_images.py --csv mnist_train.csv --out data/train --limit 20000
```

---

## 5. Loading CIFAR-10 (classic batches) without external DL libs

Use this loader to write CIFAR images into the folder structure expected above. Save as `cifar10_to_images.py`:

```python
# cifar10_to_images.py
import os, pickle, argparse
import numpy as np
import cv2

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_cifar_batch(batch_file, out_dir, start_idx=0):
    d = unpickle(batch_file)
    data = d[b'data']  # (10000, 3072)
    labels = d[b'labels']
    for i in range(data.shape[0]):
        array = data[i]
        r = array[0:1024].reshape(32,32)
        g = array[1024:2048].reshape(32,32)
        b = array[2048:].reshape(32,32)
        img = np.dstack([b,g,r])  # OpenCV BGR
        lbl = str(labels[i])
        outdir = os.path.join(out_dir, lbl)
        os.makedirs(outdir, exist_ok=True)
        fname = os.path.join(outdir, f"{lbl}_{start_idx+i:06d}.png")
        cv2.imwrite(fname, img)
    return data.shape[0]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="directory containing data_batch_1 .. data_batch_5 and test_batch")
    ap.add_argument("--out_dir", default="data/cifar_images")
    args = ap.parse_args()
    idx = 0
    for i in range(1,6):
        batch = os.path.join(args.data_dir, f"data_batch_{i}")
        n = save_cifar_batch(batch, os.path.join(args.out_dir, "train"), idx)
        idx += n
    # test batch
    save_cifar_batch(os.path.join(args.data_dir, "test_batch"), os.path.join(args.out_dir, "test"), 0)
    print("Done.")
```

Run after downloading and unpacking CIFAR-10 into `cifar-10-batches-py/`:

```bash
python cifar10_to_images.py --data_dir cifar-10-batches-py --out_dir data/cifar_images
```

---

## 6. Bag of Visual Words (BoW) — building a codebook (SIFT + KMeans)

**Overview:**

1. Detect SIFT descriptors on many images (sample N images).
2. Collect descriptors into a large array (M × 128), subsample if too big.
3. Use `sklearn.cluster.KMeans` to cluster into `k` visual words (vocabulary).
4. For each image, assign each descriptor to nearest cluster (word) and build normalized histogram (k-dim).
5. Use these histograms as feature vectors for a classifier (SVM / RandomForest).

**Script to build vocabulary:** `bow_build_vocab.py`

```python
# bow_build_vocab.py
import os, glob, argparse, random
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
import joblib

def collect_descriptors(image_paths, max_images=500, max_desc=20000):
    sift = cv2.SIFT_create()
    descriptors = []
    sampled = random.sample(image_paths, min(len(image_paths), max_images))
    for p in sampled:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        kp, des = sift.detectAndCompute(img, None)
        if des is None: continue
        descriptors.append(des)
    if len(descriptors) == 0:
        return None
    all_desc = np.vstack(descriptors)
    if all_desc.shape[0] > max_desc:
        idx = np.random.choice(all_desc.shape[0], max_desc, replace=False)
        all_desc = all_desc[idx]
    return all_desc

def build_vocab(descs, k=500, out_file="vocab_kmeans.pkl"):
    print("Clustering", descs.shape[0], "descriptors into", k, "words.")
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=10000, verbose=True)
    kmeans.fit(descs)
    joblib.dump(kmeans, out_file)
    print("Saved vocabulary to", out_file)
    return kmeans

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--k", type=int, default=500)
    ap.add_argument("--out", default="vocab_kmeans.pkl")
    args = ap.parse_args()

    # collect all image file paths (assumes folders for each class under images_dir)
    image_paths = glob.glob(os.path.join(args.images_dir, "*", "*.png")) + \
                  glob.glob(os.path.join(args.images_dir, "*", "*.jpg"))
    descs = collect_descriptors(image_paths, max_images=800, max_desc=30000)
    if descs is None:
        raise RuntimeError("No descriptors collected.")
    build_vocab(descs, k=args.k, out_file=args.out)
```

**How to use the vocabulary later:**

* Load `kmeans = joblib.load("vocab_kmeans.pkl")`.
* For each image, detect SIFT descriptors and do `words = kmeans.predict(descriptors)`.
* Build a histogram: `hist, _ = np.histogram(words, bins=range(k+1))` and L1 or L2 normalize.

You can then feed these histograms to any sklearn classifier (SVM, DecisionTree, RandomForest).

---

## 7. Example A — MNIST classification using Decision Tree (train.py / test.py)

We treat MNIST as *image files* in `data/train/<label>/img.png` and `data/test/...`.

### `mnist_train.py`

```python
# mnist_train.py
import os, glob, argparse
import numpy as np
import cv2
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

def load_dataset(folder):
    X=[]; y=[]
    classes = sorted(os.listdir(folder))
    for c in classes:
        cdir = os.path.join(folder, c)
        if not os.path.isdir(cdir): continue
        for imgf in glob.glob(os.path.join(cdir,"*.png")):
            img = cv2.imread(imgf, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            X.append(img.flatten())
            y.append(int(c))
    return np.array(X), np.array(y)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="data/mnist/train")
    ap.add_argument("--model_out", default="mnist_dt.joblib")
    args = ap.parse_args()

    X_train, y_train = load_dataset(args.train_dir)
    print("Train:", X_train.shape, y_train.shape)
    clf = DecisionTreeClassifier(max_depth=30)
    clf.fit(X_train, y_train)
    joblib.dump(clf, args.model_out)
    print("Saved model to", args.model_out)
```

### `mnist_test.py`

```python
# mnist_test.py
import os, glob, argparse
import numpy as np
import cv2
import joblib
from sklearn.metrics import accuracy_score, classification_report

def load_dataset(folder):
    X=[]; y=[]
    for c in sorted(os.listdir(folder)):
        cdir = os.path.join(folder, c)
        if not os.path.isdir(cdir): continue
        for imgf in glob.glob(os.path.join(cdir,"*.png")):
            img = cv2.imread(imgf, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            X.append(img.flatten())
            y.append(int(c))
    return np.array(X), np.array(y)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", default="data/mnist/test")
    ap.add_argument("--model", default="mnist_dt.joblib")
    args = ap.parse_args()

    X_test, y_test = load_dataset(args.test_dir)
    clf = joblib.load(args.model)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test,y_pred))
    print(classification_report(y_test,y_pred))
```

**Notes:**

* Decision trees on raw pixels are **baseline** and not state-of-the-art, but good for learning. You may try RandomForest or SVM or HOG features instead for better accuracy.

---

## 8. Example B — CIFAR-10: HOG features + SVM (train.py / test.py)

We extract HOG descriptors from color images (convert to grayscale first) using `skimage` (or OpenCV gradients + custom HOG). To avoid extra deps, we implement HOG using `skimage.feature.hog` — if `skimage` is not available, use OpenCV's `cv2.HOGDescriptor` (but cv2 HOG expects 64×128). Here we use `skimage`.

Install `scikit-image` if you want: `pip install scikit-image`.

### `cifar_hog_train.py`

```python
# cifar_hog_train.py
import os, glob, argparse
import numpy as np
import cv2
import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

def load_images_and_labels(folder, max_per_class=None):
    X=[]; y=[]
    classes = sorted(os.listdir(folder))
    for c in classes:
        cdir = os.path.join(folder, c)
        if not os.path.isdir(cdir): continue
        files = glob.glob(os.path.join(cdir,"*.png"))
        if max_per_class is not None:
            files = files[:max_per_class]
        for f in files:
            img = cv2.imread(f)
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # resize if needed (CIFAR is 32x32)
            h,w = gray.shape
            if h != 32 or w != 32:
                gray = cv2.resize(gray, (32,32))
            # HOG: tune parameters
            feat = hog(gray, orientations=9, pixels_per_cell=(8,8),
                       cells_per_block=(2,2), block_norm='L2-Hys', feature_vector=True)
            X.append(feat)
            y.append(int(c))
    return np.array(X), np.array(y)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="data/cifar_images/train")
    ap.add_argument("--model_out", default="cifar_hog_svm.joblib")
    ap.add_argument("--max_per_class", type=int, default=None)
    args = ap.parse_args()

    X_train, y_train = load_images_and_labels(args.train_dir, args.max_per_class)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    clf = LinearSVC(C=1.0, max_iter=5000)
    clf.fit(Xs, y_train)
    joblib.dump((clf, scaler), args.model_out)
    print("Saved model + scaler to", args.model_out)
```

### `cifar_hog_test.py`

```python
# cifar_hog_test.py
import os, glob, argparse
import numpy as np
import cv2
import joblib
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report

def load_images_and_labels(folder, max_per_class=None):
    X=[]; y=[]
    classes = sorted(os.listdir(folder))
    for c in classes:
        cdir = os.path.join(folder, c)
        if not os.path.isdir(cdir): continue
        files = glob.glob(os.path.join(cdir,"*.png"))
        if max_per_class is not None:
            files = files[:max_per_class]
        for f in files:
            img = cv2.imread(f)
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if gray.shape != (32,32):
                gray = cv2.resize(gray, (32,32))
            feat = hog(gray, orientations=9, pixels_per_cell=(8,8),
                       cells_per_block=(2,2), block_norm='L2-Hys', feature_vector=True)
            X.append(feat); y.append(int(c))
    return np.array(X), np.array(y)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", default="data/cifar_images/test")
    ap.add_argument("--model", default="cifar_hog_svm.joblib")
    args = ap.parse_args()

    X_test, y_test = load_images_and_labels(args.test_dir)
    clf, scaler = joblib.load(args.model)
    Xs = scaler.transform(X_test)
    y_pred = clf.predict(Xs)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
```

**Notes:**

* `skimage.feature.hog` returns feature vectors. You can tune `pixels_per_cell` and `cells_per_block` for better performance.
* `LinearSVC` is faster for large datasets. You can also use `SVC(kernel='rbf')` for non-linear decision boundary (slower).

---

## 9. Additional Example: Building BoW Features & Classifier

Once you have `vocab_kmeans.pkl` from `bow_build_vocab.py`, use this snippet to compute BoW histograms for all images and train a classifier (example SVM).

```python
# bow_train.py (sketch)
import os, glob, joblib
import numpy as np
import cv2
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

kmeans = joblib.load("vocab_kmeans.pkl")
k = kmeans.n_clusters
sift = cv2.SIFT_create()

def image_bow_hist(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    kp, des = sift.detectAndCompute(img, None)
    if des is None or len(des)==0:
        return np.zeros(k)
    words = kmeans.predict(des)
    hist, _ = np.histogram(words, bins=np.arange(k+1))
    hist = hist.astype(float) / (hist.sum()+1e-6)
    return hist

# build X,y for training folder
# then scale and train SVM as earlier
```

---

## 10. Student Tasks (Assignments)

1. **Implement and run the two pipelines:**

   * MNIST → Decision Tree (`mnist_train.py` / `mnist_test.py`)
   * CIFAR-10 → HOG → SVM (`cifar_hog_train.py` / `cifar_hog_test.py`)
   * Build BoW vocabulary and train classifier on a small dataset (optional).

2. **Swap feature detectors and classifiers:**

   * Try `ORB`, `AKAZE` descriptors instead of SIFT (OpenCV has them).
   * Replace DecisionTree with `RandomForest`, `SVM`, or `kNN`.
   * Report results (accuracy, confusion matrix) on test data and discuss which combinations performed better and why.

3. **Hyperparameter experiments:**

   * For BoW: vary `k` (vocabulary size) and show effect.
   * For HOG: tune `pixels_per_cell` and `cells_per_block`.
   * For classifiers: tune `C` and kernel for SVM, `max_depth` for DecisionTree.

4. **Data loader task**: Students must write their dataloader matching the folder format above, ensure consistent image sizes and color handling.

---

## 11. Practical tips & troubleshooting

* **SIFT**: Use `opencv-contrib-python` to get SIFT. If unavailable, use `ORB` (binary descriptors) but note BoW with ORB uses different distance metrics.
* **Performance**: KMeans on many descriptors can be slow; use `MiniBatchKMeans` and subsample descriptors.
* **Feature scaling**: Always standardize features for SVM (use `StandardScaler`).
* **Memory**: Be cautious stacking all descriptors—subsample or use streaming clustering.
* **Evaluation**: Use confusion matrix to inspect per-class errors.

---

## 12. References & further reading

* BoW tutorial: [https://www.pinecone.io/learn/series/image-search/bag-of-visual-words/](https://www.pinecone.io/learn/series/image-search/bag-of-visual-words/)
* HOG tutorial: [https://www.geeksforgeeks.org/machine-learning/hog-feature-visualization-in-python-using-skimage/](https://www.geeksforgeeks.org/machine-learning/hog-feature-visualization-in-python-using-skimage/)
* CIFAR-10: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
* MNIST CSV (Kaggle): [https://www.kaggle.com/datasets/oddrationale/mnist-in-csv](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

---
