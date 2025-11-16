# Digital Image Processing: Compression & Filtering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)

A comprehensive educational project exploring fundamental digital image processing algorithms, including JPEG compression with DCT transformation and advanced image filtering techniques for noise reduction. This hands-on project demonstrates practical implementations of compression and filtering methods using Python.

## 📚 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Topics Covered](#topics-covered)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Learning Objectives](#learning-objectives)
- [License](#license)
- [Author](#author)

## 🎯 Overview

This project provides a hands-on, interactive approach to understanding digital image processing through Jupyter notebooks. It combines theoretical implementations with practical exercises, allowing learners to implement JPEG compression algorithms and various image filtering techniques from scratch using Python.

**Course:** Digital Image Processing  
**Department:** Mathematical Sciences  
**Semester:** Spring 2025

## ✨ Features

- **Interactive Learning:** Jupyter notebook format with executable code cells and detailed explanations
- **From-Scratch Implementation:** Build DCT, quantization, and filters using NumPy
- **Real Image Processing:** Work with actual images (Lena, real-world photos)
- **Visualization Focus:** Compare original vs. processed images side-by-side
- **Quality Metrics:** MSE, PSNR, SSIM calculations for compression evaluation
- **Filter Comparison:** Analyze different kernel sizes and filter types

## 📖 Topics Covered

## 📖 Topics Covered

### 1. JPEG Compression Implementation
- **Discrete Cosine Transform (DCT):** Custom 1D and 2D DCT implementation from scratch
- **Quantization:** JPEG standard quantization matrix application
- **Zigzag Scanning:** Efficient coefficient ordering for compression
- **Image Reconstruction:** Inverse DCT for decompression
- **Quality Metrics:** MSE, PSNR, SSIM, and compression ratio calculations

### 2. Image Filtering Techniques
- **Noise Generation:** Salt and pepper noise addition with configurable intensity
- **Mean Filter:** Smoothing filter implementation (both custom and OpenCV)
- **Median Filter:** Non-linear filter for noise reduction (both custom and OpenCV)
- **Performance Analysis:** Compare filter effectiveness across kernel sizes (3×3, 5×5, 7×7)
- **Built-in vs Custom:** Validate implementations against OpenCV functions

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/koushamoeini/dip-compression-filtering.git
   cd dip-compression-filtering
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages:**
   ```bash
   pip install numpy matplotlib opencv-python scikit-image imageio jupyter
   ```

   Or install from requirements file:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

5. **Open a notebook:**
   Navigate to `notebooks/jpeg_compression.ipynb` or `notebooks/image_filtering.ipynb` in your browser

## 💻 Usage

### JPEG Compression Notebook

1. **Navigate to the notebook:** Open `notebooks/jpeg_compression.ipynb`
2. **Follow the steps:**
   - Load and preprocess images
   - Apply DCT transformation to 8×8 blocks
   - Quantize DCT coefficients
   - Convert to compressed vector format
   - Reconstruct image using inverse DCT
   - Evaluate compression quality

**Example:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Run complete JPEG pipeline
original, reconstructed, compressed_vector = run_jpeg_on_image('../assets/images/realImage.jpg')

# Evaluate quality
mse, psnr, ssim, ratio = evaluate(original, reconstructed, compressed_vector)
print(f"PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}, Compression Ratio: {ratio:.2f}")
```

### Image Filtering Notebook

1. **Navigate to the notebook:** Open `notebooks/image_filtering.ipynb`
2. **Follow the steps:**
   - Load test image (Lena)
   - Generate salt & pepper noise at different rates
   - Apply mean and median filters
   - Compare custom vs. OpenCV implementations
   - Analyze filter performance

**Example:**
```python
import cv2
import numpy as np

# Load image
gray = cv2.imread("../assets/images/lena.png", 0)

# Add noise
noisy_image = sp(gray, 0.1)  # 10% salt & pepper noise

# Apply filters
mean_filtered = apply_mean_filter(noisy_image, kernel_size=5)
median_filtered = apply_median_filter(noisy_image, kernel_size=5)

# Compare results
display_results(noisy_image, apply_mean_filter, filter_name="mean")
```

## 📁 Project Structure

```
dip-compression-filtering/
│
├── notebooks/
│   ├── jpeg_compression.ipynb       # JPEG compression implementation
│   └── image_filtering.ipynb        # Image filtering techniques
│
├── assets/
│   └── images/
│       ├── lena.png                 # Test image for filtering
│       └── realImage.jpg            # Test image for compression
│
├── src/                             # (Optional) Modular Python scripts
├── docs/                            # Additional documentation
├── .gitignore                       # Git ignore rules
├── LICENSE                          # MIT License
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 📦 Requirements

- **numpy** - Numerical computing and array operations
- **matplotlib** - Data visualization and image display
- **opencv-python (cv2)** - Image processing and built-in filters
- **scikit-image** - SSIM metric and image transformations
- **imageio** - Image I/O operations
- **jupyter** - Interactive notebook environment

## 🎓 Learning Objectives

By the end of this project, you will:

- ✅ Understand JPEG compression pipeline from theory to implementation
- ✅ Implement DCT transformation from scratch using NumPy
- ✅ Apply quantization for lossy compression
- ✅ Generate and analyze salt & pepper noise
- ✅ Build custom mean and median filters without libraries
- ✅ Compare filter performance across different kernel sizes
- ✅ Calculate image quality metrics (MSE, PSNR, SSIM)
- ✅ Validate custom implementations against industry-standard libraries
- ✅ Visualize and interpret compression artifacts and filtering effects

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Kousha Moeini**
- Email: koushamoeini@gmail.com
- GitHub: [@koushamoeini](https://github.com/koushamoeini)

---
