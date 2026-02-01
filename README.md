# 🧠 CNN Kernels Visualization Playground

An interactive web application built with **Streamlit** and **OpenCV** that allows users to visualize how Convolutional Neural Network (CNN) kernels (filters) process images in real-time.

---

## 🚀 Overview
Ever wondered how a computer "sees" an image? This playground demonstrates the mathematics of **Image Convolution**, a core building block of Computer Vision and AI. By sliding different "kernels" (weight matrices) over pixels, the app extracts features like edges, textures, and shapes—mimicking the early layers of a CNN.

### 🖥️ App Preview
<img width="1919" height="915" alt="image" src="https://github.com/user-attachments/assets/2a83dcba-c68e-462b-9e99-f208f39894e9" />
<img width="1919" height="872" alt="image" src="https://github.com/user-attachments/assets/f0c44e27-69c1-49b8-8db1-9063d5150dba" />


---

## ✨ Features
* **Real-time Convolution:** Adjust kernel intensity and see the result instantly.
* **Pre-processing:** Toggle **Grayscale Mode** to see how reducing data channels simplifies feature detection.
* **Kernel Library:** Explore 10+ different filters, including:
    * **Edge Detection (Sobel/Laplacian):** Identifies boundaries and outlines.
    * **Blurring (Gaussian/Box):** Removes noise for better feature generalization.
    * **Sharpening:** Enhances fine details and textures.
* **Weight Visualization:** View the actual mathematical matrix being applied to the image.
* **Instant Download:** Save your processed "Feature Maps" directly to your device.

---

## 🛠️ Tech Stack
* **Python:** Core logic and matrix manipulation.
* **Streamlit:** Web interface and interactivity.
* **OpenCV:** High-performance image processing.
* **NumPy:** Mathematical array operations.

---

## 🏃 Local Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/harshit-singh-hs/CNN-Kernels-Visualization.git](https://github.com/harshit-singh-hs/CNN-Kernels-Visualization.git)
    cd CNN-Kernels-Visualization
    ```

2.  **Install dependencies:**
    ```bash
    pip install streamlit opencv-python-headless numpy Pillow
    ```

3.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

---

## 🧮 How it Works: The CNN Connection
In a real Neural Network, these kernel values are **learned** during training. In this app, you manually select filters to see how they highlight specific features:

1.  **Input:** The raw image data (RGB or Grayscale).
2.  **Kernel:** A sliding window ($3 \times 3$, $5 \times 5$, etc.) that performs element-wise multiplication.
3.  **Feature Map:** The output
