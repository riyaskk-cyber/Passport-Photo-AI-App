import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from huggingface_hub import hf_hub_download # 🔹 NEW IMPORT

# Import the U2NET model architecture from your model.py file
from model import U2NET


class PassportSegmentationInference:
    def __init__(self, device=None): # 🔹 'model_path' removed from constructor
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model() # 🔹 'model_path' removed from function call
        self.transform = self.get_transform()
        print(f"Inference running on: {self.device}")

    def load_model(self): # 🔹 'model_path' removed from this method
        # 🔹 NEW HUGGING FACE DOWNLOAD LOGIC
        REPO_ID = "kkriyas/u2net-finetuned"
        FILENAME = "u2net_finetuned.pth"
        
        print("📥 Downloading model weights from Hugging Face Hub...")
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

        model = U2NET(3, 1)  # 3 input channels, 1 output channel
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print(f"✅ Successfully loaded model from: {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None

        model.to(self.device)
        model.eval()
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
            raise ValueError(f"Could not load image from {image_path}")
        original_image = image.copy()
        original_size = (image.shape[1], image.shape[0]) # (width, height)
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
        """Morphological operations to remove speckles"""
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
        binary_mask, probability_mask = self.postprocess_mask(prediction, original_size, threshold=0.7)
        binary_mask = self.clean_mask(binary_mask)
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        return {
            'original_image': original_rgb,
            'binary_mask': binary_mask,
            'probability_mask': probability_mask,
            'prediction_tensor': prediction
        }

    def apply_background(self, image, mask, mode="white"):
        mask_3ch = np.stack([mask, mask, mask], axis=2).astype(np.uint8)
        if mode == "white":
            bg = np.full_like(image, (255, 255, 255), dtype=np.uint8)
        elif mode == "blue":
            bg = np.full_like(image, (0, 0, 255), dtype=np.uint8)
        elif mode == "transparent":
            result = np.dstack((image, mask * 255))
            return result
        else:
            bg = np.full_like(image, (255, 255, 255), dtype=np.uint8)
        result = image * mask_3ch + bg * (1 - mask_3ch)
        return result.astype(np.uint8)

    def resize_passport(self, image, mode="passport"):
        if mode == "passport":
            size = (600, 600)  # 2x2 inch @ 300 DPI
        elif mode == "visa":
            size = (413, 531)  # 35x45 mm @ 300 DPI
        else:
            size = (600, 600)
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    def save_results(self, results, output_dir, base_name):
        os.makedirs(output_dir, exist_ok=True)
        mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
        cv2.imwrite(mask_path, results['binary_mask'] * 255)

        prob_path = os.path.join(output_dir, f"{base_name}_probability.png")
        cv2.imwrite(prob_path, (results['probability_mask'] * 255).astype(np.uint8))

        # Background variants
        white_bg = self.apply_background(results['original_image'], results['binary_mask'], "white")
        blue_bg = self.apply_background(results['original_image'], results['binary_mask'], "blue")
        trans_bg = self.apply_background(results['original_image'], results['binary_mask'], "transparent")

        # Resize
        passport_img = self.resize_passport(white_bg, "passport")
        visa_img = self.resize_passport(blue_bg, "visa")

        # Save outputs
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_passport.jpg"), cv2.cvtColor(passport_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_visa.jpg"), cv2.cvtColor(visa_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_transparent.png"), trans_bg)

        print(f"\n✅ Results saved to {output_dir}")
        print(f" - Binary mask: {mask_path}")
        print(f" - Probability mask: {prob_path}")
        print(f" - Passport: {base_name}_passport.jpg")
        print(f" - Visa: {base_name}_visa.jpg")
        print(f" - Transparent: {base_name}_transparent.png")


def main():
    MODEL_PATH = "u2net_finetuned.pth"
    TEST_IMAGE_PATH = "sample.jpg"
    OUTPUT_DIR = "inference_results"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inferencer = PassportSegmentationInference() # 🔹 'MODEL_PATH' removed from this line

    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"⚠️ Test image not found: {TEST_IMAGE_PATH}")
        return

    try:
        print(f"🔍 Running inference on: {TEST_IMAGE_PATH}")
        results = inferencer.predict_single_image(TEST_IMAGE_PATH)
        base_name = os.path.splitext(os.path.basename(TEST_IMAGE_PATH))[0]
        inferencer.save_results(results, OUTPUT_DIR, base_name)

        mask_coverage = np.sum(results['binary_mask']) / results['binary_mask'].size
        print(f"\n📊 Mask Statistics:")
        print(f" - Foreground coverage: {mask_coverage*100:.2f}%")
        print(f" - Image size: {results['original_image'].shape[:2]}")
        print(f" - Mask size: {results['binary_mask'].shape}")

    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
