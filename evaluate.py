"""
Evaluation script for anomaly detection model.

Generates:
    1. AUROC score (standard metric for anomaly detection)
    2. Visualization of reconstructions and error heatmaps
    3. Per-defect-type performance breakdown

Usage:
    python evaluate.py --checkpoint results/bottle_xxx/best_model.pth
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from tqdm import tqdm

from models import ConvAutoencoder
from utils import MVTecDataset


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model args
    args = checkpoint.get('args', {})
    latent_dim = args.get('latent_dim', 256)
    
    # Create and load model
    model = ConvAutoencoder(in_channels=3, latent_dim=latent_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Training loss: {checkpoint.get('train_loss', 'N/A'):.6f}")
    
    return model, args


def compute_auroc(model, test_loader, device):
    """
    Compute Area Under ROC Curve.
    
    AUROC = 1.0 is perfect, 0.5 is random chance.
    """
    all_labels = []
    all_scores = []
    all_defect_types = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Computing scores"):
            images = batch['image'].to(device)
            labels = batch['label'].numpy()
            defect_types = batch['defect_type']
            
            # Anomaly score = reconstruction error
            scores = model.get_reconstruction_error(images, per_pixel=False)
            scores = scores.cpu().numpy()
            
            all_labels.extend(labels)
            all_scores.extend(scores)
            all_defect_types.extend(defect_types)
    
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # Compute AUROC
    auroc = roc_auc_score(all_labels, all_scores)
    
    # Per-defect breakdown
    defect_results = {}
    unique_defects = set(all_defect_types)
    
    for defect in unique_defects:
        mask = np.array([d == defect for d in all_defect_types])
        defect_scores = all_scores[mask]
        defect_labels = all_labels[mask]
        
        defect_results[defect] = {
            'count': len(defect_scores),
            'mean_score': defect_scores.mean(),
            'is_anomaly': defect_labels[0] if len(defect_labels) > 0 else 0
        }
    
    return auroc, all_labels, all_scores, defect_results


def plot_roc_curve(labels, scores, save_path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = roc_auc_score(labels, scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUROC = {auroc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Anomaly Detection', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved ROC curve to {save_path}")


def visualize_reconstructions(model, test_dataset, device, save_dir, n_samples=8):
    """
    Visualize original images, reconstructions, and error heatmaps.
    
    Shows both normal and anomalous samples for comparison.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect normal and anomaly samples
    normal_indices = [i for i, s in enumerate(test_dataset.labels) if s == 0]
    anomaly_indices = [i for i, s in enumerate(test_dataset.labels) if s == 1]
    
    # Sample indices
    n_each = n_samples // 2
    selected_normal = normal_indices[:n_each] if len(normal_indices) >= n_each else normal_indices
    selected_anomaly = anomaly_indices[:n_each] if len(anomaly_indices) >= n_each else anomaly_indices
    selected = selected_normal + selected_anomaly
    
    fig, axes = plt.subplots(len(selected), 4, figsize=(16, 4 * len(selected)))
    
    if len(selected) == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(selected):
            sample = test_dataset[sample_idx]
            image = sample['image'].unsqueeze(0).to(device)
            label = sample['label']
            mask = sample['mask']
            defect_type = sample['defect_type']
            
            # Get reconstruction and error
            recon = model(image)
            error = model.get_reconstruction_error(image, per_pixel=True)
            
            # Convert to numpy for plotting
            img_np = denormalize(image[0].cpu())
            recon_np = denormalize(recon[0].cpu())
            error_np = error[0, 0].cpu().numpy()
            mask_np = mask[0].numpy()
            
            # Plot
            axes[idx, 0].imshow(img_np)
            axes[idx, 0].set_title(f'Original ({defect_type})', fontsize=10)
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(recon_np)
            axes[idx, 1].set_title('Reconstruction', fontsize=10)
            axes[idx, 1].axis('off')
            
            im = axes[idx, 2].imshow(error_np, cmap='hot')
            axes[idx, 2].set_title(f'Error Map (score: {error_np.mean():.4f})', fontsize=10)
            axes[idx, 2].axis('off')
            plt.colorbar(im, ax=axes[idx, 2], fraction=0.046)
            
            axes[idx, 3].imshow(mask_np, cmap='gray')
            axes[idx, 3].set_title('Ground Truth', fontsize=10)
            axes[idx, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'reconstructions.png', dpi=150)
    plt.close()
    print(f"Saved reconstructions to {save_dir / 'reconstructions.png'}")


def denormalize(tensor):
    """Convert from [-1,1] back to [0,1] for visualization."""
    tensor = tensor * 0.5 + 0.5  # [-1,1] -> [0,1]
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.permute(1, 2, 0).numpy()  # CHW -> HWC


def plot_score_distribution(labels, scores, defect_results, save_path):
    """Plot distribution of anomaly scores."""
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(normal_scores, bins=30, alpha=0.7, label=f'Normal (n={len(normal_scores)})', color='green')
    plt.hist(anomaly_scores, bins=30, alpha=0.7, label=f'Anomaly (n={len(anomaly_scores)})', color='red')
    
    plt.xlabel('Reconstruction Error (Anomaly Score)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Score Distribution: Normal vs Anomaly', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved score distribution to {save_path}")


def evaluate(args):
    """Main evaluation function."""
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    model, train_args = load_model(checkpoint_path, device)
    
    # Get category and data dir from checkpoint or args
    category = args.category or train_args.get('category', 'synthetic')
    data_dir = args.data_dir or train_args.get('data_dir', './data')
    image_size = train_args.get('image_size', 256)
    
    print(f"\nEvaluating on category: {category}")
    
    # Load test dataset
    test_dataset = MVTecDataset(
        root_dir=data_dir,
        category=category,
        split='test',
        image_size=image_size
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Setup output directory
    output_dir = checkpoint_path.parent / 'evaluation'
    output_dir.mkdir(exist_ok=True)
    
    # Compute AUROC
    print("\nComputing metrics...")
    auroc, labels, scores, defect_results = compute_auroc(model, test_loader, device)
    
    print(f"\n{'='*50}")
    print(f"AUROC: {auroc:.4f}")
    print(f"{'='*50}")
    
    # Per-defect breakdown
    print("\nPer-defect-type breakdown:")
    print("-" * 40)
    for defect, results in sorted(defect_results.items()):
        status = "ANOMALY" if results['is_anomaly'] else "NORMAL"
        print(f"  {defect:20s} | {status:7s} | n={results['count']:3d} | mean_score={results['mean_score']:.4f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_roc_curve(labels, scores, output_dir / 'roc_curve.png')
    plot_score_distribution(labels, scores, defect_results, output_dir / 'score_distribution.png')
    visualize_reconstructions(model, test_dataset, device, output_dir, n_samples=8)
    
    # Save results summary
    with open(output_dir / 'results.txt', 'w') as f:
        f.write(f"AUROC: {auroc:.4f}\n\n")
        f.write("Per-defect breakdown:\n")
        for defect, results in sorted(defect_results.items()):
            status = "ANOMALY" if results['is_anomaly'] else "NORMAL"
            f.write(f"  {defect}: {status}, n={results['count']}, mean_score={results['mean_score']:.4f}\n")
    
    print(f"\nResults saved to: {output_dir}")
    
    return auroc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate anomaly detection model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--category', type=str, default=None,
                        help='Dataset category (default: from checkpoint)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to dataset (default: from checkpoint)')
    
    args = parser.parse_args()
    evaluate(args)