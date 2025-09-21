# 📸 Passport / Visa Photo Generator

An AI-powered app for **real-time background removal and passport/visa photo generation**.  
Built as a capstone project for the Data Science & Machine Learning course.  

---

## ✨ Features
- 🔹 Real-time **person segmentation** using a fine-tuned **U²-Net** model  
- 🔹 Background replacement:
  - White (passport photos)  
  - Blue (visa/ID photos)  
  - Transparent (custom use)  
- 🔹 Automatic cropping to standard sizes:
  - Passport (600×600 px = 2×2 inch)  
  - Visa (413×531 px = 35×45 mm)  
- 🔹 Supports **camera input** (mobile friendly) or **file upload**  
- 🔹 Download ready-to-use photo in JPG/PNG format  

---

## 📊 Model & Evaluation
- **Base model:** [U²-Net](https://github.com/xuebinqin/U-2-Net)  
- **Training dataset:** ~200 manually labeled images (via CVAT)  
- **Metrics (validation set):**
  - Mean IoU: `0.9258`  
  - Mean Dice: `0.9611`  
  - Mean Accuracy: `0.9557`  

---

## 🖥️ Tech Stack
- **Frontend:** Streamlit  
- **Model Framework:** PyTorch  
- **Image Processing:** OpenCV, Albumentations, Pillow  
- **Deployment:** Streamlit Cloud (works on desktop & mobile browsers)  

---

## 🚀 Deployment
Live app 👉 [**Demo App**](https://passport-photo-ai-app-m5zss3y6tpi3abo4knghif.streamlit.app/)  

⚠️ **Note about model weights:**  
The trained model (`u2net_finetuned.pth`) is stored on **Hugging Face Hub** instead of GitHub (due to size limits).  
It will be **automatically downloaded** at runtime using the `huggingface-hub` library.  

---

## ⚙️ Installation (Local Setup)
Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/passport-Photo-AI-app.git
cd passport-photo-app
pip install -r requirements.txt
streamlit run app.py
