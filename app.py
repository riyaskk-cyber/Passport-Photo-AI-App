import os
import streamlit as st
import numpy as np
import cv2
import torch
from huggingface_hub import hf_hub_download

from model import U2NET
from inference import PassportSegmentationInference

# ------------------ CONFIG ------------------
MODEL_REPO = "kkriyas/u2net-finetuned"
MODEL_FILE = "u2net_finetuned.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Download model from Hugging Face if missing
MODEL_PATH = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)

@st.cache_resource
def load_inferencer():
    return PassportSegmentationInference(MODEL_PATH, device=DEVICE)

inferencer = load_inferencer()

# ------------------ UTILITY FUNCTION FOR SIMPLE CROPPING ------------------
def simple_center_crop(image, output_size):
    """
    Crops the image from the center to the specified output size.
    """
    h, w, _ = image.shape
    target_w, target_h = output_size
    
    # Calculate the starting point for the crop
    start_x = max(0, w // 2 - target_w // 2)
    start_y = max(0, h // 2 - target_h // 2)
    
    # Calculate the ending point
    end_x = start_x + target_w
    end_y = start_y + target_h
    
    # Make sure the crop window doesn't go out of bounds
    if end_x > w:
        start_x -= (end_x - w)
        end_x = w
    if end_y > h:
        start_y -= (end_y - h)
        end_y = h
    
    cropped_img = image[start_y:end_y, start_x:end_x]
    
    # Resize to the final output size
    resized_img = cv2.resize(cropped_img, output_size, interpolation=cv2.INTER_AREA)
    
    return resized_img

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

# ------------------ DEMO ------------------
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

        # Apply background
        if bg_option == "White":
            out_img_bgr = inferencer.apply_background(results["original_image"], results["binary_mask"], "white")
        elif bg_option == "Blue":
            out_img_bgr = inferencer.apply_background(results["original_image"], results["binary_mask"], "blue")
        else:
            out_img_bgr = inferencer.apply_background(results["original_image"], results["binary_mask"], "transparent")
            
        # Determine the final output size based on user choice
        final_size = (600, 600) if crop_option.startswith("Passport") else (413, 531)
        
        # Apply the simple center cropping and resizing
        final_img = simple_center_crop(out_img_bgr, final_size)

    st.success("✅ Done! Your photo is ready.")

    # Show preview (smaller)
    st.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB) if bg_option != "Transparent" else final_img, caption="Processed Image", width=300)

    # Save for download
    if bg_option == "Transparent":
        dl_file = "output.png"
        cv2.imwrite(dl_file, final_img)
    else:
        dl_file = "output.jpg"
        cv2.imwrite(dl_file, final_img)

    with open(dl_file, "rb") as f:
        st.download_button(
            label="⬇️ Download Final Photo",
            data=f,
            file_name=dl_file,
            mime="image/png" if bg_option == "Transparent" else "image/jpeg"
        )
