"""
Download MVTec Anomaly Detection dataset.

Option 1: Manual download from Kaggle (recommended - most reliable)
Option 2: Use kagglehub to download programmatically
Option 3: Direct download from MVTec (may require registration)

Dataset info: https://www.mvtec.com/company/research/datasets/mvtec-ad
"""

import os
import shutil
from pathlib import Path


def setup_from_kaggle_manual(kaggle_download_path: str, data_dir: str = './data'):
    """
    Setup dataset from manual Kaggle download.
    
    Instructions:
    1. Go to: https://www.kaggle.com/datasets/ipythonx/mvtec-ad
    2. Click 'Download' (you'll need a Kaggle account)
    3. Extract the downloaded zip file
    4. Call this function with the path to extracted folder
    
    Args:
        kaggle_download_path: Path to extracted Kaggle download
        data_dir: Where to set up the data
    """
    src = Path(kaggle_download_path)
    dst = Path(data_dir)
    
    if not src.exists():
        raise FileNotFoundError(f"Source path not found: {src}")
    
    dst.mkdir(parents=True, exist_ok=True)
    
    # Copy categories
    categories = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]
    
    for cat in categories:
        cat_src = src / cat
        cat_dst = dst / cat
        if cat_src.exists() and not cat_dst.exists():
            print(f"Copying {cat}...")
            shutil.copytree(cat_src, cat_dst)
            print(f"  Done: {cat}")
    
    print(f"\nDataset ready at: {dst.absolute()}")


def download_with_kagglehub(data_dir: str = './data'):
    """
    Download using kagglehub (pip install kagglehub).
    Requires Kaggle API credentials.
    
    Setup credentials:
    1. Go to kaggle.com -> Account -> Create API Token
    2. Save kaggle.json to ~/.kaggle/kaggle.json
    """
    try:
        import kagglehub
    except ImportError:
        print("Installing kagglehub...")
        os.system("pip install kagglehub")
        import kagglehub
    
    print("Downloading MVTec AD from Kaggle...")
    print("Note: You need Kaggle API credentials (~/.kaggle/kaggle.json)")
    
    # Download dataset
    path = kagglehub.dataset_download("ipythonx/mvtec-ad")
    print(f"Downloaded to: {path}")
    
    # Setup in our data directory
    setup_from_kaggle_manual(path, data_dir)
    
    return path


def create_synthetic_test_data(data_dir: str = './data', category: str = 'synthetic'):
    """
    Create synthetic test data for development/testing.
    Useful if you can't download the real dataset immediately.
    
    Creates simple geometric shapes with synthetic "defects".
    """
    import numpy as np
    from PIL import Image, ImageDraw
    
    data_path = Path(data_dir) / category
    train_path = data_path / 'train' / 'good'
    test_good_path = data_path / 'test' / 'good'
    test_defect_path = data_path / 'test' / 'defect'
    gt_path = data_path / 'ground_truth' / 'defect'
    
    for p in [train_path, test_good_path, test_defect_path, gt_path]:
        p.mkdir(parents=True, exist_ok=True)
    
    img_size = 256
    
    def create_normal_image(seed):
        """Create a 'normal' image - clean circle on gradient background."""
        np.random.seed(seed)
        
        # Gradient background
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        for i in range(img_size):
            img[i, :, :] = [50 + i//4, 50 + i//4, 60 + i//4]
        
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        
        # Draw a clean circle
        center = img_size // 2
        radius = 60 + np.random.randint(-10, 10)
        draw.ellipse(
            [center - radius, center - radius, center + radius, center + radius],
            fill=(200, 200, 210),
            outline=(150, 150, 160),
            width=3
        )
        
        return img
    
    def create_defect_image(seed):
        """Create a 'defective' image with a scratch/spot."""
        img = create_normal_image(seed)
        draw = ImageDraw.Draw(img)
        
        np.random.seed(seed + 1000)
        
        # Create mask
        mask = Image.new('L', (img_size, img_size), 0)
        mask_draw = ImageDraw.Draw(mask)
        
        # Add a defect (scratch or spot)
        if np.random.random() > 0.5:
            # Scratch
            x1 = np.random.randint(80, 180)
            y1 = np.random.randint(80, 180)
            x2 = x1 + np.random.randint(-40, 40)
            y2 = y1 + np.random.randint(-40, 40)
            draw.line([(x1, y1), (x2, y2)], fill=(50, 50, 50), width=3)
            mask_draw.line([(x1, y1), (x2, y2)], fill=255, width=5)
        else:
            # Spot
            cx = np.random.randint(100, 156)
            cy = np.random.randint(100, 156)
            r = np.random.randint(5, 15)
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(30, 30, 30))
            mask_draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=255)
        
        return img, mask
    
    # Generate training images (normal only)
    print("Creating synthetic training images...")
    for i in range(50):
        img = create_normal_image(i)
        img.save(train_path / f'{i:03d}.png')
    
    # Generate test images - normal
    print("Creating synthetic test images (normal)...")
    for i in range(10):
        img = create_normal_image(i + 100)
        img.save(test_good_path / f'{i:03d}.png')
    
    # Generate test images - defective
    print("Creating synthetic test images (defective)...")
    for i in range(20):
        img, mask = create_defect_image(i + 200)
        img.save(test_defect_path / f'{i:03d}.png')
        mask.save(gt_path / f'{i:03d}_mask.png')
    
    print(f"\nSynthetic dataset created at: {data_path.absolute()}")
    print(f"  Training (normal): {len(list(train_path.glob('*.png')))} images")
    print(f"  Test (normal): {len(list(test_good_path.glob('*.png')))} images")
    print(f"  Test (defect): {len(list(test_defect_path.glob('*.png')))} images")
    
    return data_path


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup MVTec AD dataset')
    parser.add_argument('--method', type=str, default='synthetic',
                        choices=['synthetic', 'kagglehub', 'manual'],
                        help='Download method')
    parser.add_argument('--kaggle-path', type=str, default=None,
                        help='Path to manually downloaded Kaggle data')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory to save data')
    parser.add_argument('--category', type=str, default='synthetic',
                        help='Category name for synthetic data')
    
    args = parser.parse_args()
    
    if args.method == 'synthetic':
        print("Creating synthetic test dataset...")
        print("(Use --method kagglehub or --method manual for real data)\n")
        create_synthetic_test_data(args.data_dir, args.category)
        
    elif args.method == 'kagglehub':
        download_with_kagglehub(args.data_dir)
        
    elif args.method == 'manual':
        if not args.kaggle_path:
            print("Manual setup instructions:")
            print("1. Go to: https://www.kaggle.com/datasets/ipythonx/mvtec-ad")
            print("2. Download and extract the dataset")
            print("3. Run: python utils/download_data.py --method manual --kaggle-path /path/to/extracted/folder")
        else:
            setup_from_kaggle_manual(args.kaggle_path, args.data_dir)