import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import U2NET
from huggingface_hub import hf_hub_download


class PassportSegmentationInference:
    def __init__(self, model_path=None, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Download from Hugging Face if path not provided
        if model_path is None or not os.path.exists(model_path):
            model_path = hf_hub_download(repo_id="kkriyas/u2net-finetuned", filename="u2net_finetuned.pth")

        self.model = self.load_model(model_path)
        self.transform = self.get_transform()
        print(f"Inference running on: {self.device}")

    def load_model(self, model_path):
        """Load U2NET architecture and apply saved weights (state_dict)."""
        model = U2NET(3, 1)  # 3 input channels, 1 output channel
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        print(f"✅ Successfully loaded model weights from {model_path}")
        return model

    def get_transform(self):
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            ToTensorV2(),
        ])

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        original_image = image.copy()
        original_size = (image.shape[1], image.shape[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0)
        return image_tensor, original_image, original_size

    def postprocess_mask(self, prediction, original_size, threshold=0.6):
        mask = torch.sigmoid(prediction).squeeze().cpu().numpy()
        mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_LINEAR)
        binary_mask = (mask_resized > threshold).astype(np.uint8)
        return binary_mask, mask_resized

    def clean_mask(self, mask, kernel_size=5):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        return cleaned

    def predict_single_image(self, image_path):
        image_tensor, original_image, original_size = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            prediction = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        binary_mask, probability_mask = self.postprocess_mask(prediction, original_size, threshold=0.7)
        binary_mask = self.clean_mask(binary_mask)
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        return {
            "original_image": original_rgb,
            "binary_mask": binary_mask,
            "probability_mask": probability_mask,
            "prediction_tensor": prediction
        }

    def apply_background(self, image, mask, mode="white"):
        mask_3ch = np.stack([mask, mask, mask], axis=2).astype(np.uint8)
        if mode == "white":
            bg = np.full_like(image, (255, 255, 255), dtype=np.uint8)
        elif mode == "blue":
            bg = np.full_like(image, (0, 0, 255), dtype=np.uint8)
        elif mode == "transparent":
            return np.dstack((image, mask * 255))
        else:
            bg = np.full_like(image, (255, 255, 255), dtype=np.uint8)
        return image * mask_3ch + bg * (1 - mask_3ch)

    def resize_passport(self, image, mode="passport"):
        if mode == "passport":
            size = (600, 600)
        elif mode == "visa":
            size = (413, 531)
        else:
            size = (600, 600)
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
