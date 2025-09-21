import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Absolute paths
image_folder = '/content/drive/MyDrive/AI_Passport_Project/Project/data/photo'
mask_folder = '/content/drive/MyDrive/AI_Passport_Project/Project/data/SegmentationClass/mask'

class SegmentationDataset(Dataset):
    def __init__(self, image_folder, mask_folder, transform=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_filenames = sorted(os.listdir(image_folder))
        self.mask_filenames = sorted(os.listdir(mask_folder))
        self.transform = transform
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_folder, self.mask_filenames[idx])
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)  # ensure binary
        mask = (mask / 255).astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
            # Ensure image is float32 and normalized to [0,1]
            if image.dtype != torch.float32:
                image = image.float() / 255.0
                
            # Add channel dimension to mask if needed
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)  # Add channel dim (1, H, W)
        else:
            # Convert to tensors manually if no transform
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).float().unsqueeze(0)
            
        return image, mask

def get_dataloaders(batch_size=8, val_split=0.2):
    # Define augmentations
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),  # This ensures [0,1] range
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),  # This ensures [0,1] range
        ToTensorV2(),
    ])
    
    # Create separate datasets for train and val with their respective transforms
    train_dataset = SegmentationDataset(image_folder, mask_folder, transform=train_transform)
    val_dataset = SegmentationDataset(image_folder, mask_folder, transform=val_transform)
    
    # Get total size for splitting
    total_size = len(train_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    # Create indices for splitting
    indices = list(range(total_size))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subset datasets
    from torch.utils.data import Subset
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Total samples: {total_size}")
    print(f"Train samples: {train_size}, Validation samples: {val_size}")
    
    return train_loader, val_loader

# Test script
if __name__ == '__main__':
    train_loader, val_loader = get_dataloaders(batch_size=4)
    
    # Test train loader
    sample_img, sample_mask = next(iter(train_loader))
    print("Train Image batch shape:", sample_img.shape)  # [B, 3, 256, 256]
    print("Train Image dtype:", sample_img.dtype)
    print("Train Image value range:", sample_img.min().item(), "to", sample_img.max().item())
    print("Train Mask batch shape:", sample_mask.shape)  # [B, 1, 256, 256]
    print("Train Unique mask values:", torch.unique(sample_mask))
    
    # Test val loader
    val_img, val_mask = next(iter(val_loader))
    print("\nVal Image batch shape:", val_img.shape)
    print("Val Image dtype:", val_img.dtype)
    print("Val Image value range:", val_img.min().item(), "to", val_img.max().item())
    print("Val Mask batch shape:", val_mask.shape)
    print("Val Unique mask values:", torch.unique(val_mask))