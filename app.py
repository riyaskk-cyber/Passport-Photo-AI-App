import os
import streamlit as st
import numpy as np
import cv2
import torch
import gdown
from model import U2NET
from inference import PassportSegmentationInference

# ------------------ CONFIG ------------------
MODEL_PATH = "u2net_finetuned.pth"   # local path for weights
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Auto-download model from Google Drive if missing
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1VMWHakp4PDcSWqIMu_cGm930xJEZ3e59"  # 🔹 replace with your actual file ID
    st.write("📥 Downloading model weights...")
    gdown.download(url, MODEL_PATH, quiet=False)

@st.cache_resource
def load_inferencer():
    return PassportSegmentationInference(MODEL_PATH, device=DEVICE)

inferencer = load_inferencer()

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Passport Photo Generator", layout="wide")

# ------------------ CUSTOM STYLING ------------------
st.markdown(
    """
    <style>
    body {
        background-color: #F7FBFF;
    }
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.05);
    }
    h1 {
        color: #2E86C1;
        text-align: center;
        font-family: 'Segoe UI', sans-serif;
    }
    h3 {
        color: #1B4F72;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton button {
        background-color: #2E86C1;
        color: white;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-size: 1em;
        border: none;
    }
    .stButton button:hover {
        background-color: #1B4F72;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ HEADER ------------------
st.markdown(
    """
    <h1>📸 Passport / Visa Photo Generator</h1>
    <p style="text-align:center; color:gray; font-size:16px;">
        Upload or capture → AI segmentation → Background replacement → Passport-ready download
    </p>
    """,
    unsafe_allow_html=True,
)

# ------------------ DEMO TAB ------------------
st.markdown("### Generate Passport/ID Photos")

# Input method selection
choice = st.radio("Choose input method:", ["Upload", "Camera"], horizontal=True)
img_source = None

if choice == "Upload":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img_source = uploaded_file
elif choice == "Camera":
    camera_file = st.camera_input("Take a photo")
    if camera_file:
        img_source = camera_file

if img_source:
    # Read image
    file_bytes = np.asarray(bytearray(img_source.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    tmp_path = "temp_input.jpg"
    cv2.imwrite(tmp_path, image)

    # Options
    bg_option = st.radio("Background", ["White", "Blue", "Transparent"], horizontal=True)
    crop_option = st.radio("Output Size", ["Passport (600x600)", "Visa (413x531)"], horizontal=True)

    # Run model with spinner
    with st.spinner("⏳ Processing your photo..."):
        results = inferencer.predict_single_image(tmp_path)

        if bg_option == "White":
            out_img = inferencer.apply_background(results["original_image"], results["binary_mask"], "white")
        elif bg_option == "Blue":
            out_img = inferencer.apply_background(results["original_image"], results["binary_mask"], "blue")
        else:
            out_img = inferencer.apply_background(results["original_image"], results["binary_mask"], "transparent")

        if crop_option.startswith("Passport"):
            out_img = inferencer.resize_passport(out_img, "passport")
        else:
            out_img = inferencer.resize_passport(out_img, "visa")

    st.success("✅ Done! Your photo is ready.")

    # Show preview (smaller)
    st.image(out_img, caption="Processed Image", width=300)

    # Save for download
    if bg_option == "Transparent":
        dl_file = "output.png"
        cv2.imwrite(dl_file, out_img)
    else:
        dl_file = "output.jpg"
        cv2.imwrite(dl_file, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

    with open(dl_file, "rb") as f:
        st.download_button(
            label="⬇️ Download Final Photo",
            data=f,
            file_name=dl_file,
            mime="image/png" if bg_option == "Transparent" else "image/jpeg"
        )
