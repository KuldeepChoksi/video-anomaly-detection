"""
Training script for anomaly detection autoencoder.

Key idea: Train ONLY on normal images. The model learns to reconstruct
normal patterns. At test time, anomalies have high reconstruction error.

Usage:
    python train.py --category synthetic --epochs 50
    python train.py --category bottle --epochs 100
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import ConvAutoencoder
from utils import MVTecDataset


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train for one epoch.
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        images = batch['image'].to(device)
        
        # Forward pass
        reconstructions = model(images)
        loss = criterion(reconstructions, images)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """
    Validate model on test set.
    
    Returns:
        avg_loss: Average reconstruction loss
        avg_normal_error: Average error on normal images
        avg_anomaly_error: Average error on anomaly images
    """
    model.eval()
    total_loss = 0
    normal_errors = []
    anomaly_errors = []
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['label']
            
            # Get reconstruction and error
            reconstructions = model(images)
            loss = criterion(reconstructions, images)
            total_loss += loss.item()
            
            # Per-image error for analysis
            errors = model.get_reconstruction_error(images, per_pixel=False)
            
            for err, label in zip(errors.cpu().numpy(), labels.numpy()):
                if label == 0:
                    normal_errors.append(err)
                else:
                    anomaly_errors.append(err)
    
    avg_loss = total_loss / len(val_loader)
    avg_normal = sum(normal_errors) / len(normal_errors) if normal_errors else 0
    avg_anomaly = sum(anomaly_errors) / len(anomaly_errors) if anomaly_errors else 0
    
    return avg_loss, avg_normal, avg_anomaly


def train(args):
    """Main training function."""
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Create datasets
    print(f"\nLoading dataset: {args.category}")
    train_dataset = MVTecDataset(
        root_dir=args.data_dir,
        category=args.category,
        split='train',
        image_size=args.image_size
    )
    
    test_dataset = MVTecDataset(
        root_dir=args.data_dir,
        category=args.category,
        split='test',
        image_size=args.image_size
    )
    
    print(f"Training samples: {len(train_dataset)} (all normal)")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = ConvAutoencoder(in_channels=3, latent_dim=args.latent_dim)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Learning rate scheduler - reduce LR when loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Setup results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.results_dir) / f"{args.category}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 60)
    
    best_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'normal_err': [], 'anomaly_err': []}
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, normal_err, anomaly_err = validate(model, test_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['normal_err'].append(normal_err)
        history['anomaly_err'].append(anomaly_err)
        
        # Calculate separation ratio (higher is better)
        separation = anomaly_err / normal_err if normal_err > 0 else 0
        
        # Print progress
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Normal Err: {normal_err:.6f} | "
              f"Anomaly Err: {anomaly_err:.6f} | "
              f"Separation: {separation:.2f}x")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args)
            }, results_dir / 'best_model.pth')
            print(f"  → Saved best model (loss: {val_loss:.6f})")
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'args': vars(args)
    }, results_dir / 'final_model.pth')
    
    print("-" * 60)
    print(f"Training complete!")
    print(f"Best validation loss: {best_loss:.6f}")
    print(f"Final separation ratio: {history['anomaly_err'][-1] / history['normal_err'][-1]:.2f}x")
    print(f"Models saved to: {results_dir}")
    
    return model, history, results_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train anomaly detection model')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Path to dataset')
    parser.add_argument('--category', type=str, default='synthetic',
                        help='Dataset category (e.g., bottle, synthetic)')
    parser.add_argument('--image-size', type=int, default=256,
                        help='Input image size')
    
    # Model arguments
    parser.add_argument('--latent-dim', type=int, default=256,
                        help='Latent space dimension')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers')
    
    # Output arguments
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    train(args)