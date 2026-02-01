import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

# ---------------- Page Config -----------------
st.set_page_config(
    page_title="CNN Kernel Playground",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .stDownloadButton { display: flex; justify-content: center; }
    .theory-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f0f2f6;
        border-left: 5px solid #ff4b4b;
        color: #31333F;
        margin: 10px 0px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Kernel Descriptions -----------------
EXPLANATIONS = {
    "Original": "No kernel applied. This is the 'Input Layer' of the network.",
    "Box Blur": "Averages all pixels in a neighborhood. This removes noise so the model doesn't get distracted by irrelevant details.",
    "Gaussian Blur": "A 'smarter' blur that preserves edges better than a standard Box Blur by weighting the center more heavily.",
    "Sharpen": "Exaggerates differences between pixels to find fine patterns and textures.",
    "Edge Detection (Laplacian)": "The 'Outline' filter. It calculates where brightness changes most rapidly in any direction.",
    "Sobel X": "Specifically detects Vertical Edges. Great for identifying upright structures.",
    "Sobel Y": "Specifically detects Horizontal Edges. Great for identifying floor lines or horizons.",
    "Sobel Magnitude": "Combines X and Y directions to find all strong borders in the image.",
    "Emboss": "Highlights highlights and shadows to create a 3D effect, helping the AI understand depth.",
    "Motion Blur": "Simulates movement, teaching a model to recognize objects even when they are not perfectly still."
}

# ---------------- Sidebar -----------------
with st.sidebar:
    st.header("🕹️ Controls")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    filter_type = st.selectbox("Choose Filter", list(EXPLANATIONS.keys()))
    strength = st.slider("Filter Strength", 1, 15, 3)
    
    st.divider()
    st.subheader("🧪 CNN Training Concepts")
    use_gray = st.checkbox("Grayscale Mode", value=False)
    show_kernel = st.checkbox("Show Weight Matrix", value=True)

# ---------------- Helper Functions -----------------
def apply_filter(img, filter_type, strength):
    start = time.time()
    kernel = None
    
    # Kernel Logic
    if filter_type == "Original":
        result = img.copy().astype(np.float32)
        kernel = np.array([[0,0,0],[0,1,0],[0,0,0]])
    elif filter_type == "Box Blur":
        k = 2 * strength + 1
        kernel = np.ones((k, k), np.float32) / (k * k)
        result = cv2.filter2D(img, -1, kernel)
    elif filter_type == "Gaussian Blur":
        k = 2 * strength + 1
        result = cv2.GaussianBlur(img, (k, k), strength).astype(np.float32)
        g_kernel = cv2.getGaussianKernel(k, strength)
        kernel = g_kernel @ g_kernel.T
    elif filter_type == "Sharpen":
        kernel = np.array([[0, -1, 0], [-1, 4 + strength, -1], [0, -1, 0]], dtype=np.float32)
        result = cv2.filter2D(img, -1, kernel)
    elif filter_type == "Edge Detection (Laplacian)":
        kernel = strength * np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
        result = cv2.filter2D(img, -1, kernel)
    elif filter_type == "Sobel X":
        kernel = strength * np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
        result = cv2.filter2D(img, -1, kernel)
    elif filter_type == "Sobel Y":
        kernel = strength * np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)
        result = cv2.filter2D(img, -1, kernel)
    elif filter_type == "Sobel Magnitude":
        kx = strength * np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
        ky = strength * np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)
        gx = cv2.filter2D(img, cv2.CV_32F, kx)
        gy = cv2.filter2D(img, cv2.CV_32F, ky)
        result = cv2.magnitude(gx, gy)
        kernel = {"Sobel X": kx, "Sobel Y": ky}
    elif filter_type == "Emboss":
        kernel = strength * np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
        result = cv2.filter2D(img, -1, kernel) + 128
    elif filter_type == "Motion Blur":
        k = 2 * strength + 1
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0
        kernel /= kernel.sum()
        result = cv2.filter2D(img, -1, kernel)

    
    result_final = np.clip(result, 0, 255).astype(np.uint8)
    elapsed = (time.time() - start) * 1000
    return result_final, kernel, elapsed

# ---------------- UI Layout -----------------
st.markdown("## 🔥 Image Convolution Playground")
st.markdown("### *Learn how kernels process visual data*")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    if use_gray:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    processed, kernel, elapsed = apply_filter(img_np, filter_type, strength)

    # Explanation Section
    st.markdown(f"""<div class="theory-box"><b>💡 Feature Logic ({filter_type}):</b><br>{EXPLANATIONS[filter_type]}</div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Input")
        st.image(img_np, use_container_width=True)
    with col2:
        st.subheader("✨ Processed Feature")
        st.image(processed, use_container_width=True)
        
        # Download in BGR fix
        download_ready = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR) if len(processed.shape)==3 else processed
        st.download_button("💾 Save Result", cv2.imencode('.png', download_ready)[1].tobytes(), f"{filter_type}.png", "image/png", use_container_width=True)

    # Detailed CNN Theory Section
    st.divider()
    t1, t2 = st.tabs(["🧮 Kernel Math", "🌑 Grayscale Logic"])
    
    with t1:
        st.markdown("### How the Kernel slides")
        st.write("This matrix is a 'Filter'. In a CNN, these numbers are **learned weights** that the AI adjusts to recognize objects.")
        
        if isinstance(kernel, dict):
            for name, k_val in kernel.items():
                st.write(f"**{name}**")
                st.dataframe(np.round(k_val, 3))
        else:
            st.dataframe(np.round(kernel, 3))
            
    with t2:
        st.markdown("### Grayscale Processing")
        st.write("Processing an image in grayscale reduces the computational load by 66% (1 channel instead of 3). Since kernels primarily look for **Contrast** and **Edges**, color is often unnecessary for the early layers of a network.")

else:
    st.info("Upload an image in the sidebar to begin experimenting!")