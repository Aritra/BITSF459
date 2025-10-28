
# Lab Sheet 8 — Introduction to PyTorch and Fully Connected Neural Networks

---

## 🎯 Objectives

In this lab, you will:

* Learn the **fundamentals of PyTorch** and setting up CUDA.
* Build and train a **fully connected neural network (FCNN)** for image classification.
* Understand how to create **datasets, dataloaders, and train/validation/test splits**.
* Visualize training progress using **TensorBoard**.
* Experiment with **hyperparameters** (hidden size, activation, dropout, batch size).
* Extend experiments to **CIFAR-10** and analyze how input ordering affects learning.

---

## ⚙️ Section 1 — Installing and Configuring PyTorch

### 1.1 Conda Setup

Create a new environment for this lab:

```bash
conda create -n torchlab python=3.10
conda activate torchlab
```

Install PyTorch and TorchVision with CUDA support (select CUDA version as per your GPU):

```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# For CPU only
pip install torch torchvision torchaudio
```

Verify installation:

```python
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

---

## 🔢 Section 2 — PyTorch Basics

### 2.1 Tensors

```python
import torch
a = torch.rand(3, 3)
b = torch.ones(3, 3)
c = a + b
print("Result:\n", c)
```

### 2.2 GPU Usage

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a = a.to(device)
print("Tensor is on:", a.device)
```

---

## 🧱 Section 3 — Building a Fully Connected Neural Network (FCNN)

### 3.1 Defining a Simple FCNN

We’ll build a 3-layer network for MNIST (input 784 → hidden → hidden → output 10).

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleFCNN(nn.Module):
    def __init__(self):
        super(SimpleFCNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)      # flatten input image
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
```

### 3.2 Concepts Introduced

* **Batch Normalization** (`nn.BatchNorm1d`) – stabilizes learning.
* **Dropout** (`nn.Dropout`) – prevents overfitting.
* **Activation Functions**:

  * `F.relu(x)` – default for hidden layers.
  * `F.leaky_relu(x, 0.1)` – to avoid dead neurons.
  * `torch.sigmoid(x)` or `F.tanh(x)` – for experiments.

---

## 📦 Section 4 — Loading Data and Creating Dataloaders

### 4.1 Download MNIST Dataset

We’ll reuse the MNIST CSV or TorchVision dataset.

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)
```

---

## 🧮 Section 5 — Training the Model

### 5.1 Optimizer and Loss

Once your model and dataloaders are ready, the next step is to define:

1. The **loss function** — how the model’s predictions are penalized when they deviate from true labels.
2. The **optimizer** — how the model’s parameters are updated based on gradients computed from the loss.

---

#### 🎯 **Common Loss Functions in Classification**

| Loss Function           | Description                                      | Typical Use                                       |
| ----------------------- | ------------------------------------------------ | ------------------------------------------------- |
| `nn.CrossEntropyLoss()` | Combines softmax + negative log-likelihood.      | Multiclass classification (e.g. MNIST, CIFAR-10). |
| `nn.BCELoss()`          | Binary Cross Entropy (expects sigmoid output).   | Binary classification tasks.                      |
| `nn.MSELoss()`          | Mean Squared Error.                              | Regression or autoencoder reconstruction.         |
| `nn.NLLLoss()`          | Negative Log-Likelihood (used with log-softmax). | Alternative to CrossEntropyLoss.                  |

**Example:**

```python
criterion = nn.CrossEntropyLoss()
```

---

#### ⚙️ **Optimizers in PyTorch**

Optimizers control *how weights are updated* during training.
You can easily switch optimizers with one line of code.

| Optimizer             | Key Characteristics                                   | Typical Hyperparameters              |
| --------------------- | ----------------------------------------------------- | ------------------------------------ |
| `torch.optim.SGD`     | Basic stochastic gradient descent.                    | `lr`, `momentum`, `weight_decay`     |
| `torch.optim.Adam`    | Adaptive learning rates per parameter (most popular). | `lr`, `betas`, `eps`, `weight_decay` |
| `torch.optim.RMSprop` | Scales updates by recent gradient magnitudes.         | `lr`, `alpha`, `momentum`            |
| `torch.optim.Adagrad` | Accumulates squared gradients to adapt learning rate. | `lr`, `lr_decay`                     |
| `torch.optim.AdamW`   | Variant of Adam with better weight decay behavior.    | `lr`, `betas`, `weight_decay`        |

**Example Setup:**

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
```

---

#### 🔁 **Dynamic Learning Rate Adjustment**

Learning rate scheduling helps models converge faster and avoid local minima.
You can adjust learning rates **manually** or use **schedulers** provided by PyTorch.

| Scheduler           | Description                                         | Typical Use                  |
| ------------------- | --------------------------------------------------- | ---------------------------- |
| `StepLR`            | Reduces learning rate by a factor every few epochs. | Long training schedules.     |
| `ReduceLROnPlateau` | Reduces LR when validation loss stops improving.    | Adaptive training.           |
| `ExponentialLR`     | Decays LR exponentially after each epoch.           | Continuous LR decay.         |
| `CosineAnnealingLR` | Smoothly varies LR using cosine function.           | For cyclical learning rates. |

**Example with StepLR:**

```python
from torch.optim.lr_scheduler import StepLR

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # halve LR every 5 epochs

for epoch in range(20):
    train_one_epoch()
    scheduler.step()
    print(f"Epoch {epoch+1}: Learning Rate = {scheduler.get_last_lr()}")
```

**Example with ReduceLROnPlateau (adaptive):**

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

for epoch in range(20):
    val_loss = validate_model()
    scheduler.step(val_loss)
```

---

#### 📈 **Inspecting and Updating Learning Rate Manually**

```python
for param_group in optimizer.param_groups:
    print(param_group['lr'])   # check current learning rate
    param_group['lr'] = 0.0005  # change dynamically
```

---

#### 💡 **Best Practices**

* Start with **Adam** (`lr=0.001`) — most stable for beginners.
* For fine-tuning or small datasets — use **SGD with momentum**.
* Always monitor learning rate using TensorBoard or logs.
* If the model stops improving → consider **reducing LR** dynamically.
* Combine with **weight decay** to prevent overfitting on small datasets.

---


### 5.2 Training Loop

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="runs/mnist_fcnn")

for epoch in range(10):
    model.train()
    total_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    print(f"Epoch {epoch+1}: Training Loss = {avg_loss:.4f}")

writer.close()
```

---

## 📊 Section 6 — Using TensorBoard for Visualization

Install TensorBoard:

```bash
pip install tensorboard
```

Run it in terminal:

```bash
tensorboard --logdir=runs
```

Then open [http://localhost:6006](http://localhost:6006) in your browser to see:

* **Training vs Validation loss**
* **Accuracy curves**
* **Hyperparameter comparisons**

---

## 🧠 Section 7 — Experimentation (MNIST)

Students must:

* Vary **hidden layer sizes** (e.g., 128, 512).
* Try different **activation functions** (ReLU, LeakyReLU, Tanh).
* Adjust **dropout values** (0.1–0.5).
* Modify **batch size** (32, 64, 128).
* Observe convergence, accuracy, and loss curves via TensorBoard.

### Example Hyperparameter Test:

| Hidden sizes | Activation | Dropout | Batch size | Accuracy |
| ------------ | ---------- | ------- | ---------- | -------- |
| 256-128      | ReLU       | 0.3     | 64         | 97.2%    |
| 512-256      | LeakyReLU  | 0.5     | 128        | 98.1%    |

---

## 🧩 Section 8 — CIFAR-10 Experiments (Flattened Input)

Now extend the same concept to **CIFAR-10** (32×32×3 input → 3072 features).

### 8.1 Load CIFAR-10

```python
from torchvision import datasets, transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)
```

### 8.2 Network Definition

Modify the input layer to 3072 (32×32×3):

```python
class CIFARFCNN(nn.Module):
    def __init__(self):
        super(CIFARFCNN, self).__init__()
        self.fc1 = nn.Linear(3072, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
```

---

## 🧮 Section 9 — CIFAR-10 Input Variants

### Task 1: Interleaved Channels (RGBRGB…)

Flatten as alternating color values per pixel.

```python
x = img.permute(1,2,0).reshape(-1)  # row-major interleaving R,G,B per pixel
```

### Task 2: Channel Concatenation (All R, then G, then B)

```python
x = torch.cat([img[0].flatten(), img[1].flatten(), img[2].flatten()])
```

### Task 3: Row-Level Interleaving (RowR, RowG, RowB)

```python
rows = []
for i in range(img.shape[1]):
    rows.append(img[0,i,:])  # row from R
    rows.append(img[1,i,:])  # row from G
    rows.append(img[2,i,:])  # row from B
x = torch.cat(rows)
```

---

## 📈 Section 10 — Analysis

Students must:

* Train the same FCNN with all **three input organizations**.
* Record and compare:

  * Training/validation loss curves.
  * Final test accuracy.
* Analyze **why channel or row order affects performance** — discuss **neighborhood information** and **spatial coherence**.

### Expected Insight:

* Pixel-level interleaving preserves **local color context**.
* Channel concatenation destroys neighborhood structure → lower performance.
* Row-level interleaving falls in between.

---

## 🧾 Deliverables

1. `mnist_fcnn_train.py` — code for MNIST FCNN.
2. `cifar_fcnn_experiments.py` — CIFAR-10 experiments for all 3 input variants.
3. TensorBoard logs (`runs/` folder) with loss/accuracy curves.
4. Comparative report:

   * Table of results (accuracy, convergence rate).
   * Observations on activation, dropout, and input layout effects.

---

## 📚 References

* PyTorch Tutorials: [https://pytorch.org/tutorials](https://pytorch.org/tutorials)
* CIFAR-10 Dataset: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
* MNIST Dataset: [https://www.kaggle.com/datasets/oddrationale/mnist-in-csv](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
* TensorBoard for PyTorch: [https://pytorch.org/docs/stable/tensorboard.html](https://pytorch.org/docs/stable/tensorboard.html)

---
