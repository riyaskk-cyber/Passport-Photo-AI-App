import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm

from data_loader import get_dataloaders
from model import U2NET  # u2net.py renamed as model.py

# ===== 1. Model Loader =====
def get_model(pretrained_path=None, device="cpu"):
    """
    Load the U2NET model, optionally with pretrained weights.
    
    This function will load a full checkpoint, including the optimizer state,
    and move everything to the correct device.
    """
    model = U2NET(3, 1)  # 3 channels input, 1 output (binary mask)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    start_epoch, best_iou = 0, 0.0
    
    if pretrained_path and os.path.exists(pretrained_path):
        try:
            # Load the full checkpoint
            checkpoint = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # --- The new fix ---
            # Move optimizer tensors to the correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            # --- End of fix ---
            
            start_epoch = checkpoint['epoch'] + 1
            best_iou = checkpoint.get('best_iou', 0.0)
            print(f"Loaded checkpoint and resuming from epoch {start_epoch-1}. Best IoU so far: {best_iou:.4f}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
            print("Training from scratch...")
    else:
        print("No pretrained weights found. Training from scratch...")
        
    return model, optimizer, start_epoch, best_iou

# ===== 2. Dice Loss Function (NEW) =====
def dice_loss(pred, target, smooth=1e-6):
    """
    A loss function based on the Dice coefficient, which is great for
    semantic segmentation and helps with class imbalance.
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice_coeff = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice_coeff

# ===== 3. Combined Loss Function (NEW) =====
class CombinedLoss(nn.Module):
    """
    A combination of Binary Cross-Entropy and Dice Loss.
    This helps the model learn both the pixel-level classification (BCE)
    and the overall shape of the object (Dice Loss), leading to cleaner masks.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss_val = dice_loss(torch.sigmoid(pred), target)
        
        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss_val)

# ===== 4. Metrics =====
def calculate_metrics(outputs, masks):
    # Ensure outputs and masks are float32
    outputs = outputs.float()
    masks = masks.float()
    
    preds = (torch.sigmoid(outputs) > 0.5).float()

    intersection = (preds * masks).sum()
    union = preds.sum() + masks.sum() - intersection
    iou = (intersection / (union + 1e-6)).item()

    correct = (preds == masks).sum().item()
    total = masks.numel()
    pixel_acc = correct / total

    return iou, pixel_acc

# ===== 5. Training Function with Validation =====
def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                num_epochs=100, checkpoint_dir="./checkpoints", metrics_file="metrics.csv",
                start_epoch=0, best_iou=0.0):

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'u2net_checkpoint.pth')
    
    # Track metrics
    metrics_history = []

    # Move model to device and ensure it's in the right precision
    model.to(device)
    model = model.float()  # Ensure model parameters are float32

    for epoch in range(start_epoch, num_epochs):
        # ===== Train =====
        model.train()
        running_loss = 0.0
        train_iou = 0.0
        train_acc = 0.0
        
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch+1}/{num_epochs}]")
        
        for batch_idx, (images, masks) in enumerate(loop):
            # Ensure data types are correct
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Handle U2NET multiple outputs (take the first one)
            if isinstance(outputs, (list, tuple)):
                main_output = outputs[0]
            else:
                main_output = outputs
            
            # Ensure output is float32
            main_output = main_output.float()
            
            # Compute loss
            loss = criterion(main_output, masks) # NOW USES THE NEW COMBINED LOSS
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            running_loss += loss.item()
            
            # Calculate training metrics
            with torch.no_grad():
                iou, acc = calculate_metrics(main_output, masks)
                train_iou += iou
                train_acc += acc
            
            loop.set_postfix(loss=loss.item(), iou=iou)

        epoch_loss = running_loss / len(train_loader)
        train_iou /= len(train_loader)
        train_acc /= len(train_loader)

        # ===== Validate =====
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validating", leave=False):
                # Ensure data types are correct
                images = images.to(device, dtype=torch.float32)
                masks = masks.to(device, dtype=torch.float32)
                
                outputs = model(images)
                
                # Handle U2NET multiple outputs
                if isinstance(outputs, (list, tuple)):
                    main_output = outputs[0]
                else:
                    main_output = outputs
                
                main_output = main_output.float()
                
                loss = criterion(main_output, masks)
                val_loss += loss.item()
                
                iou, acc = calculate_metrics(main_output, masks)
                val_iou += iou
                val_acc += acc

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_acc /= len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train - Loss: {epoch_loss:.4f}, IoU: {train_iou:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Acc: {val_acc:.4f}")
        print("-" * 50)

        # Save checkpoint if best IoU
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'best_loss': val_loss
            }, checkpoint_path)
            print(f"🎉 New best model saved with IoU {best_iou:.4f} at epoch {epoch+1}")

        # Save metrics for each epoch
        metrics_history.append({
            "epoch": epoch+1,
            "train_loss": epoch_loss,
            "train_iou": train_iou,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "val_acc": val_acc
        })

        # Save metrics every 10 epochs
        if (epoch + 1) % 10 == 0:
            pd.DataFrame(metrics_history).to_csv(metrics_file, index=False)

    # Save final metrics
    pd.DataFrame(metrics_history).to_csv(metrics_file, index=False)
    return model

# ===== 6. Main Script =====
if __name__ == "__main__":
    # Data
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(batch_size=4)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model and Optimizer
    # This will now load the optimizer state as well
    checkpoint_path = "./checkpoints/u2net_checkpoint.pth"
    model, optimizer, start_epoch, best_iou = get_model(checkpoint_path, device)

    # Loss
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    
    print("Training on:", device)
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name())

    # Train
    print("\nStarting training...")
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, device,
                                num_epochs=200, metrics_file="metrics.csv",
                                start_epoch=start_epoch, best_iou=best_iou)

    # Save final model state dictionary
    final_model_path = "u2net_finetuned.pth"
    torch.save(trained_model.state_dict(), final_model_path)
    print(f"Final model saved as {final_model_path}")
