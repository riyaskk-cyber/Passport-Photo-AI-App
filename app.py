import os
import streamlit as st
import numpy as np
import cv2
import torch
from model import U2NET
from inference import PassportSegmentationInference

# ------------------ CONFIG ------------------
MODEL_PATH = "u2net_finetuned.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_inferencer():
    return PassportSegmentationInference(MODEL_PATH, device=DEVICE)

inferencer = load_inferencer()

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Passport Photo Generator", layout="wide")

# ------------------ HEADER ------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#2E86C1;'>📸 Passport Photo Generator</h1>
    <p style='text-align:center; color:gray; font-size:16px;'>
        Upload or capture → AI segmentation → Background replacement → Passport-ready download
    </p>
    """,
    unsafe_allow_html=True,
)

# ------------------ DEMO ------------------
st.markdown("### Generate Passport/ID Photos")

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
    file_bytes = np.asarray(bytearray(img_source.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    tmp_path = "temp_input.jpg"
    cv2.imwrite(tmp_path, image)

    bg_option = st.radio("Background", ["White", "Blue", "Transparent"], horizontal=True)
    crop_option = st.radio("Output Size", ["Passport (600x600)", "Visa (413x531)"], horizontal=True)
    resize_mode = st.radio("Cropping Mode", ["Fit Resize", "Center Crop"], horizontal=True)

    with st.spinner("⏳ Processing your photo..."):
        results = inferencer.predict_single_image(tmp_path)

        # Background replacement
        out_img = inferencer.apply_background(
            results["original_image"],
            results["probability_mask"],   # use soft mask
            mode=bg_option.lower()
        )

        # Cropping
        if resize_mode == "Center Crop":
            h, w = out_img.shape[:2]
            min_dim = min(h, w)
            start_x = (w - min_dim) // 2
            start_y = (h - min_dim) // 2
            out_img = out_img[start_y:start_y+min_dim, start_x:start_x+min_dim]

        # Resize
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
