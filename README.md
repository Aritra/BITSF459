
# 📸 Computer Vision Demo Code Repository – BITS F459

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)]()

This repository contains **demo codes and experiments** used during lectures and labs for the **Computer Vision** course *(Subject Code: BITS F459)* at BITS Pilani Hyderabad campus.

> 📌 Refer to this repository regularly for in-class coding demos, updates, and assignment templates.

---

## 🖥️ System Requirement Notice

All students **must ensure** their system has an **Ubuntu (Linux)** partition.

- 💾 **Minimum space required**: 50 GB
- 🐧 Native Ubuntu is recommended, but you may alternatively use:
  - **WSL2** (on Windows 10/11)
  - **VirtualBox** or **VMWare** with Ubuntu

Ubuntu ensures smoother compatibility with vision toolchains and dependencies.

---

## 🐍 Python Environment Setup (Miniconda)

We’ll use **Miniconda** to manage the Python environment for all demos and assignments.

### 📥 Step 1: Install Miniconda

- Download: https://docs.conda.io/en/latest/miniconda.html
- Install it using default options
- After installation, restart terminal and verify:

```bash
conda --version
````

---

### 🛠️ Step 2: Create Environment (Recommended via YAML)

Clone this repository and run:

```bash
conda env create -f environment.yml
conda activate cv-env
```

Alternatively, create and install manually:

```bash
conda create -n cv-env python=3.10
conda activate cv-env
pip install opencv-python opencv-contrib-python numpy matplotlib open3d typing_extensions
```
You may add neccessary libraries later to your environment using pip

---

### ✅ Step 3: Verify Setup

Run this minimal test script:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

print("OpenCV version:", cv2.__version__)
```

---

## 🧑‍💻 Recommended Code Editor: VSCode

We recommend using **Visual Studio Code** for writing and debugging code.

* Download: [https://code.visualstudio.com/](https://code.visualstudio.com/)
* Install **Python extension** from the marketplace
* Configure terminal to use Conda environment


---

## 📦 Sample Environment File (`environment.yml`)

```yaml
name: cv-env
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.9
  - numpy
  - matplotlib
  - pip
  - pip:
      - opencv-python
      - opencv-contrib-python
      - open3d
      - typing_extensions
```

---

## 📄 License

This repository is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for full details.

---


