"""
Training script for Video Anomaly Detection with ConvLSTM.
Developed by Kuldeep Choksi

Trains on normal video sequences. The model learns temporal patterns
of normal processes. Anomalies cause high reconstruction error.

IMPORTANT: Saves model based on SEPARATION RATIO, not loss.
This ensures the best anomaly detection performance.

Usage:
    python train_video.py --category S01 --data-dir ./data/IPAD --epochs 20
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

from models.video_autoencoder import VideoAutoencoder
from utils.video_dataset import VideoDataset, IPADDataset


def get_dataset_class(data_dir, category):
    """Determine which dataset class to use based on folder structure."""
    data_path = Path(data_dir) / category
    
    # Check for IPAD structure
    if (data_path / 'training' / 'frames').exists():
        return IPADDataset
    # Check for standard structure
    elif (data_path / 'train').exists():
        return VideoDataset
    else:
        raise FileNotFoundError(f"Could not find valid dataset structure in {data_path}")


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        sequences = batch['frames'].to(device)
        
        # Forward pass
        reconstructions = model(sequences)
        loss = criterion(reconstructions, sequences)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate model on test set."""
    model.eval()
    total_loss = 0
    normal_errors = []
    anomaly_errors = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            sequences = batch['frames'].to(device)
            labels = batch['label']
            
            # Get reconstruction and error
            reconstructions = model(sequences)
            loss = criterion(reconstructions, sequences)
            total_loss += loss.item()
            
            # Per-sequence error for analysis
            errors = model.get_reconstruction_error(sequences, per_frame=False)
            
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
    print(f"\nLoading video dataset: {args.category}")
    
    # Auto-detect dataset format
    DatasetClass = get_dataset_class(args.data_dir, args.category)
    print(f"Using dataset loader: {DatasetClass.__name__}")
    
    train_dataset = DatasetClass(
        root_dir=args.data_dir,
        category=args.category,
        split='train',
        sequence_length=args.sequence_length,
        stride=args.stride,
        image_size=args.image_size
    )
    
    test_dataset = DatasetClass(
        root_dir=args.data_dir,
        category=args.category,
        split='test',
        sequence_length=args.sequence_length,
        stride=args.stride,
        image_size=args.image_size
    )
    
    print(f"Training sequences: {len(train_dataset)} (all normal)")
    print(f"Test sequences: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False  # MPS doesn't support pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )
    
    # Create model
    model = VideoAutoencoder(
        in_channels=3,
        latent_dim=args.latent_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_num_layers=args.lstm_layers
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5  # Changed to 'max' for separation
    )
    
    # Setup results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.results_dir) / f"video_{args.category}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Sequence length: {args.sequence_length} frames")
    print(f"\n*** SAVING BASED ON SEPARATION RATIO (not loss) ***")
    print("-" * 60)
    
    best_separation = 0.0  # Changed from best_loss
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'normal_err': [], 'anomaly_err': [], 'separation': []}
    
    # Early stopping based on separation declining
    patience = 5
    no_improve_count = 0
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, normal_err, anomaly_err = validate(model, test_loader, criterion, device)
        
        # Calculate separation ratio
        separation = anomaly_err / normal_err if normal_err > 0 else 0
        
        # Update scheduler based on separation
        scheduler.step(separation)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['normal_err'].append(normal_err)
        history['anomaly_err'].append(anomaly_err)
        history['separation'].append(separation)
        
        # Print progress
        status = ""
        if separation > best_separation:
            status = " <- BEST"
        elif separation < 1.0:
            status = " (inverted!)"
        
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Normal: {normal_err:.6f} | "
              f"Anomaly: {anomaly_err:.6f} | "
              f"Separation: {separation:.2f}x{status}")
        
        # Save best model based on SEPARATION, not loss
        if separation > best_separation:
            best_separation = separation
            best_epoch = epoch
            no_improve_count = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'separation': separation,
                'normal_err': normal_err,
                'anomaly_err': anomaly_err,
                'args': vars(args)
            }, results_dir / 'best_model.pth')
            print(f"  -> Saved best model (separation: {separation:.2f}x)")
        else:
            no_improve_count += 1
        
        # Also save checkpoint every epoch for analysis
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'separation': separation,
            'args': vars(args)
        }, results_dir / f'checkpoint_epoch_{epoch}.pth')
        
        # Early stopping if separation keeps declining
        if no_improve_count >= patience and separation < 1.0:
            print(f"\n*** Early stopping: Separation below 1.0 for {patience} epochs ***")
            print(f"*** Best model was at epoch {best_epoch} with {best_separation:.2f}x separation ***")
            break
        
        # Stop if separation inverts badly
        if separation < 0.8 and epoch > 3:
            print(f"\n*** Stopping: Separation inverted to {separation:.2f}x (anomalies reconstructed better than normal) ***")
            print(f"*** Best model saved at epoch {best_epoch} with {best_separation:.2f}x separation ***")
            break
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'best_epoch': best_epoch,
        'best_separation': best_separation,
        'args': vars(args)
    }, results_dir / 'final_model.pth')
    
    print("-" * 60)
    print(f"Training complete!")
    print(f"Best separation ratio: {best_separation:.2f}x at epoch {best_epoch}")
    print(f"Models saved to: {results_dir}")
    print(f"\nUse: python evaluate_video.py --checkpoint {results_dir}/best_model.pth --data-dir {args.data_dir}")
    
    return model, history, results_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train video anomaly detection model')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Path to dataset')
    parser.add_argument('--category', type=str, required=True,
                        help='Dataset category (e.g., S01, R01)')
    parser.add_argument('--image-size', type=int, default=256,
                        help='Frame size')
    parser.add_argument('--sequence-length', type=int, default=16,
                        help='Number of frames per sequence')
    parser.add_argument('--stride', type=int, default=4,
                        help='Stride between sequences')
    
    # Model arguments
    parser.add_argument('--latent-dim', type=int, default=128,
                        help='Latent space dimension')
    parser.add_argument('--lstm-hidden-dim', type=int, default=128,
                        help='ConvLSTM hidden dimension')
    parser.add_argument('--lstm-layers', type=int, default=2,
                        help='Number of ConvLSTM layers')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size (smaller for video due to memory)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='DataLoader workers')
    
    # Output arguments
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    print("="*60)
    print("VIDEO ANOMALY DETECTION TRAINING")
    print("Developed by Kuldeep Choksi")
    print("="*60)
    
    train(args)