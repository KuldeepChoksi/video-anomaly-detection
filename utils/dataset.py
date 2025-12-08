"""
Dataset loader for MVTec Anomaly Detection dataset.

MVTec AD contains 15 categories of industrial objects/textures.
Each category has:
    - train/ : Only normal (defect-free) images
    - test/  : Both normal and various defect types
    - ground_truth/ : Pixel-level masks showing defect locations

We train ONLY on normal images, then evaluate on both normal and defective.
"""

import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


class MVTecDataset(Dataset):
    """
    PyTorch Dataset for MVTec AD.
    
    Args:
        root_dir: Path to MVTec AD dataset
        category: Which object category (e.g., 'bottle', 'cable', 'pill')
        split: 'train' or 'test'
        transform: Optional torchvision transforms
        mask_transform: Optional transform for ground truth masks
    """
    
    # All available categories in MVTec AD (plus synthetic for testing)
    CATEGORIES = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
        'synthetic'  # For development/testing without real data
    ]
    
    def __init__(
        self, 
        root_dir: str, 
        category: str, 
        split: str = 'train',
        transform=None,
        mask_transform=None,
        image_size: int = 256
    ):
        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split
        self.image_size = image_size
        
        # Validate category
        if category not in self.CATEGORIES:
            raise ValueError(f"Category must be one of {self.CATEGORIES}")
        
        # Default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                # Normalize to [-1, 1] range (better for autoencoders)
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
            
        if mask_transform is None:
            self.mask_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ])
        else:
            self.mask_transform = mask_transform
        
        # Build list of image paths and labels
        self.images = []
        self.labels = []  # 0 = normal, 1 = anomaly
        self.masks = []   # Ground truth masks (only for test anomalies)
        self.defect_types = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Scan directory and build list of all images with labels."""
        
        split_dir = self.root_dir / self.category / self.split
        gt_dir = self.root_dir / self.category / 'ground_truth'
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Dataset not found at {split_dir}")
        
        # Each subfolder in split_dir is a defect type (or 'good' for normal)
        for defect_type in sorted(os.listdir(split_dir)):
            defect_dir = split_dir / defect_type
            
            if not defect_dir.is_dir():
                continue
                
            # Get all images in this defect folder
            for img_name in sorted(os.listdir(defect_dir)):
                if not img_name.endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                img_path = defect_dir / img_name
                self.images.append(img_path)
                self.defect_types.append(defect_type)
                
                # Label: 0 for normal ('good'), 1 for any defect
                if defect_type == 'good':
                    self.labels.append(0)
                    self.masks.append(None)
                else:
                    self.labels.append(1)
                    # Load corresponding ground truth mask
                    mask_name = img_name.replace('.png', '_mask.png')
                    mask_path = gt_dir / defect_type / mask_name
                    self.masks.append(mask_path if mask_path.exists() else None)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Transformed image tensor [C, H, W]
            label: 0 for normal, 1 for anomaly
            mask: Ground truth mask tensor [1, H, W] or zeros if no mask
        """
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Label
        label = self.labels[idx]
        
        # Load mask if available, otherwise return zeros
        mask_path = self.masks[idx]
        if mask_path is not None and mask_path.exists():
            mask = Image.open(mask_path).convert('L')  # Grayscale
            mask = self.mask_transform(mask)
        else:
            mask = torch.zeros(1, self.image_size, self.image_size)
        
        return {
            'image': image,
            'label': label,
            'mask': mask,
            'path': str(img_path),
            'defect_type': self.defect_types[idx]
        }


def get_dataloaders(
    root_dir: str,
    category: str,
    batch_size: int = 32,
    image_size: int = 256,
    num_workers: int = 4
):
    """
    Convenience function to get train and test dataloaders.
    
    Returns:
        train_loader: DataLoader with only normal images
        test_loader: DataLoader with both normal and anomalous images
    """
    
    train_dataset = MVTecDataset(
        root_dir=root_dir,
        category=category,
        split='train',
        image_size=image_size
    )
    
    test_dataset = MVTecDataset(
        root_dir=root_dir,
        category=category,
        split='test',
        image_size=image_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


# Quick test when running this file directly
if __name__ == '__main__':
    print("MVTec Dataset Loader")
    print(f"Available categories: {MVTecDataset.CATEGORIES}")