# Digital Image Processing Toolkit

A comprehensive Python implementation of fundamental digital image processing algorithms, focusing on JPEG compression and image filtering techniques. This project demonstrates practical applications of DCT-based compression, quantization, and various noise reduction filters.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

## 📋 Table of Contents

- [Features](#features)
- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Notebooks Overview](#notebooks-overview)
- [Examples](#examples)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## ✨ Features

### JPEG Compression Implementation
- **DCT (Discrete Cosine Transform)**: Custom 1D and 2D DCT implementation from scratch
- **Quantization**: Standard JPEG quantization matrix application
- **Zigzag Scanning**: Efficient coefficient ordering for compression
- **Image Reconstruction**: Inverse DCT for decompression
- **Quality Metrics**: MSE, PSNR, SSIM, and compression ratio calculations

### Image Filtering
- **Noise Generation**: Salt and pepper noise addition with configurable intensity
- **Mean Filter**: Smoothing filter implementation (custom and OpenCV)
- **Median Filter**: Non-linear filter for noise reduction (custom and OpenCV)
- **Performance Comparison**: Side-by-side analysis of different filter sizes (3×3, 5×5, 7×7)

## 🛠 Technologies

- **Python 3.7+**
- **NumPy** - Numerical computations and array operations
- **Matplotlib** - Data visualization and image display
- **OpenCV (cv2)** - Image processing and built-in filters
- **scikit-image** - SSIM metric and image transformations
- **imageio** - Image I/O operations
- **Jupyter Notebook** - Interactive development environment

## 📁 Project Structure

```
dip-compression-filtering/
├── notebooks/
│   ├── jpeg_compression.ipynb       # JPEG compression implementation
│   └── image_filtering.ipynb        # Image filtering techniques
├── assets/
│   └── images/
│       ├── lena.png                 # Test image for filtering
│       └── realImage.jpg            # Test image for compression
├── src/                             # (Optional) Modular Python scripts
├── docs/                            # Additional documentation
├── .gitignore                       # Git ignore rules
├── LICENSE                          # MIT License
└── README.md                        # This file
```

## 📦 Prerequisites

Before running the project, ensure you have:

- Python 3.7 or higher
- pip (Python package installer)
- Jupyter Notebook or JupyterLab

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/koushamoeini/dip-compression-filtering.git
   cd dip-compression-filtering
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install numpy matplotlib opencv-python scikit-image imageio jupyter
   ```

   Or create a `requirements.txt`:
   ```txt
   numpy>=1.19.0
   matplotlib>=3.3.0
   opencv-python>=4.5.0
   scikit-image>=0.18.0
   imageio>=2.9.0
   jupyter>=1.0.0
   ```
   
   Then install:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

## 💻 Usage

### JPEG Compression

Open `notebooks/jpeg_compression.ipynb` and run the cells sequentially:

```python
import numpy as np
import matplotlib.pyplot as plt

# Load and process image
from jpeg_compression import run_jpeg_on_image

original, reconstructed, compressed_vector = run_jpeg_on_image('../assets/images/realImage.jpg')

# Evaluate compression quality
mse, psnr, ssim, ratio = evaluate(original, reconstructed, compressed_vector)
print(f"PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}, Compression Ratio: {ratio:.2f}")
```

### Image Filtering

Open `notebooks/image_filtering.ipynb`:

```python
import cv2
import numpy as np

# Load image
gray = cv2.imread("../assets/images/lena.png", 0)

# Add salt and pepper noise
noisy_image = sp(gray, 0.1)  # 10% noise

# Apply mean filter
filtered = apply_mean_filter(noisy_image, kernel_size=5)

# Apply median filter
median_filtered = apply_median_filter(noisy_image, kernel_size=5)
```

## 📓 Notebooks Overview

### 1. JPEG Compression (`jpeg_compression.ipynb`)

**Key Steps:**
- Divide image into 8×8 blocks
- Apply 2D DCT to each block
- Quantize coefficients using JPEG standard matrix
- Convert to 1D vector using zigzag scanning
- Reconstruct image using inverse DCT
- Calculate quality metrics (MSE, PSNR, SSIM)

**Functions:**
- `apply_dct(image_array)` - Forward DCT transform
- `quantize(dct_blocks)` - Quantization step
- `mat_2_vec(jpeg_matrix)` - Zigzag scanning
- `decompress_from_vector(jpeg_vector)` - Full decompression pipeline
- `evaluate(original, reconstructed, jpeg_vector)` - Quality metrics

### 2. Image Filtering (`image_filtering.ipynb`)

**Key Steps:**
- Generate salt & pepper noise at various intensities
- Implement custom mean filter (NumPy only)
- Implement custom median filter (NumPy only)
- Compare with OpenCV built-in functions
- Analyze filter performance across kernel sizes

**Functions:**
- `sp(pic, n)` - Salt and pepper noise generation
- `apply_mean_filter(image, kernel_size)` - Custom mean filter
- `apply_median_filter(image, kernel_size)` - Custom median filter
- `builtin_mean_filter(image, kernel_size)` - OpenCV mean filter
- `builtin_median_filter(image, kernel_size)` - OpenCV median filter

## 📊 Examples

### JPEG Compression Results

| Metric | Value |
|--------|-------|
| MSE | ~50-100 |
| PSNR | ~30-35 dB |
| SSIM | ~0.85-0.95 |
| Compression Ratio | ~8-15x |

### Filter Comparison

**Observation:** Larger kernel sizes provide stronger noise reduction but blur fine details more significantly.

| Kernel Size | Noise Reduction | Detail Preservation |
|-------------|-----------------|---------------------|
| 3×3 | Moderate | High |
| 5×5 | Good | Moderate |
| 7×7 | Excellent | Low |

## 🖼 Results

<!-- Add screenshots here once available -->
_Screenshots will be added showing:_
- Original vs. compressed images
- Noisy images with different noise levels
- Filter comparison results

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**Commit Convention:**
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code formatting
- `refactor:` - Code restructuring
- `test:` - Test additions
- `chore:` - Configuration/dependencies

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Contact

**Kousha Moeini**
- Student ID: 401100228
- GitHub: [@koushamoeini](https://github.com/koushamoeini)
- Repository: [dip-compression-filtering](https://github.com/koushamoeini/dip-compression-filtering)

## 🙏 Acknowledgments

- Course: Digital Image Processing
- Department: Mathematical Sciences
- Semester: Spring 2025

---

⭐ If you find this project helpful, please consider giving it a star!
