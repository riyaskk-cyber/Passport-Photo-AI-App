import os
import cv2
import numpy as np
import torch
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from model import U2NET


class PassportSegmentationInference:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)
        self.transform = self.get_transform()
        print(f"Inference running on: {self.device}")

    def load_model(self, model_path):
        model = U2NET(3, 1)
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        print(f"✅ Successfully loaded model weights from {model_path}")
        return model

    def get_transform(self):
        return Compose([
            Resize(256, 256),
            Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            ToTensorV2(),
        ])

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        original_image = image.copy()
        original_size = (image.shape[1], image.shape[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0)
        return image_tensor, original_image, original_size

    def postprocess_mask(self, prediction, original_size, threshold=0.6):
        mask = torch.sigmoid(prediction).squeeze().cpu().numpy()
        mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_CUBIC)
        binary_mask = (mask_resized > threshold).astype(np.uint8)
        return binary_mask, mask_resized

    def clean_mask(self, mask, kernel_size=3):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        return cleaned

    def predict_single_image(self, image_path):
        if self.model is None:
            raise ValueError("Model not loaded properly")

        image_tensor, original_image, original_size = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device, dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            prediction = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

        binary_mask, probability_mask = self.postprocess_mask(
            prediction, original_size, threshold=0.6
        )

        binary_mask = self.clean_mask(binary_mask, kernel_size=3)
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        return {
            "original_image": original_rgb,
            "binary_mask": binary_mask,
            "probability_mask": probability_mask,
            "prediction_tensor": prediction
        }

    def apply_background(self, image, mask, mode="white"):
        if mask.dtype == np.uint8:
            soft_mask = mask.astype(np.float32)
        else:
            soft_mask = mask.copy()

        soft_mask = cv2.GaussianBlur(soft_mask, (7, 7), 0)
        soft_mask = np.clip(soft_mask, 0, 1)
        soft_mask_3ch = np.expand_dims(soft_mask, axis=2)

        if mode == "white":
            bg = np.ones_like(image, dtype=np.uint8) * 255
        elif mode == "blue":
            bg = np.zeros_like(image, dtype=np.uint8)
            bg[:] = (180, 200, 255)  # lighter blue
        elif mode == "transparent":
            alpha = (soft_mask * 255).astype(np.uint8)
            return np.dstack((image, alpha))
        else:
            bg = np.ones_like(image, dtype=np.uint8) * 255

        blended = (image * soft_mask_3ch + bg * (1 - soft_mask_3ch)).astype(np.uint8)
        return blended

    def resize_passport(self, image, mode="passport"):
        if mode == "passport":
            size = (600, 600)
        elif mode == "visa":
            size = (413, 531)
        else:
            size = (600, 600)
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    MODEL_PATH = "u2net_finetuned.pth"
    TEST_IMAGE_PATH = "sample.jpg"
    OUTPUT_DIR = "inference_results"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inferencer = PassportSegmentationInference(MODEL_PATH)

    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Test image not found: {TEST_IMAGE_PATH}")
    else:
        results = inferencer.predict_single_image(TEST_IMAGE_PATH)
        out = inferencer.apply_background(results["original_image"], results["probability_mask"], mode="white")
        out = inferencer.resize_passport(out, "passport")
        cv2.imwrite(os.path.join(OUTPUT_DIR, "sample_passport.jpg"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        print("✅ Saved improved passport photo.")
