# evaluate.py
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from data_loader import get_dataloaders
from model import U2NET

# ---------- Metric functions ----------
def dice_score(pred, target, smooth=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    inter = (pred * target).sum()
    return (2 * inter + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + smooth) / (union + smooth)

def pixel_accuracy(pred, target):
    return (pred == target).sum() / target.size

# ---------- Visualization ----------
def save_visuals(image, gt_mask, pred_mask, save_path):
    """Save side-by-side: input | GT | Pred | Composite"""
    image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    gt_mask = (gt_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
    pred_mask = (pred_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)

    # Composite with white background
    alpha = pred_mask.astype(float) / 255.0
    bg = np.ones_like(image, dtype=np.uint8) * 255
    composite = (alpha[..., None] * image + (1 - alpha[..., None]) * bg).astype(np.uint8)

    # Convert grayscale to 3-channel for stacking
    gt_bgr = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
    pred_bgr = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)

    stacked = np.hstack([image, gt_bgr, pred_bgr, composite])
    cv2.imwrite(save_path, stacked)

# ---------- Main ----------
def evaluate(model_path="u2net_finetuned.pth", batch_size=1, device="cuda"):
    os.makedirs("eval_results", exist_ok=True)

    # Load validation set
    _, val_loader = get_dataloaders(batch_size=batch_size)

    # Load model
    model = U2NET(3, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    metrics = []

    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(val_loader, desc="Evaluating")):
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            outputs = model(images)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            for b in range(images.size(0)):
                img = images[b].cpu()
                gt = masks[b].cpu()
                pr = preds[b].cpu()

                # Metrics
                iou = iou_score(pr.numpy(), gt.numpy())
                dice = dice_score(pr.numpy(), gt.numpy())
                acc = pixel_accuracy(pr.numpy(), gt.numpy())

                metrics.append({
                    "image_idx": idx * batch_size + b,
                    "IoU": iou,
                    "Dice": dice,
                    "PixelAcc": acc
                })

                # Save visualization
                save_path = os.path.join("eval_results", f"result_{idx*batch_size+b}.jpg")
                save_visuals(img, gt, pr, save_path)

    # Save metrics CSV
    df = pd.DataFrame(metrics)
    df.to_csv("eval_metrics.csv", index=False)

    print("\n=== Evaluation Summary ===")
    print(f"Mean IoU:   {df['IoU'].mean():.4f}")
    print(f"Mean Dice:  {df['Dice'].mean():.4f}")
    print(f"Mean Acc:   {df['PixelAcc'].mean():.4f}")
    print("===========================")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate(model_path="u2net_finetuned.pth", batch_size=1, device=device)
