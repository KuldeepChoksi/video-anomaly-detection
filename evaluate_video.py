"""
Evaluation script for Video Anomaly Detection.
Developed by Kuldeep Choksi

Evaluates trained video models and generates:
- AUROC score
- Per-frame anomaly scores
- Side-by-side visualization video (Original | Reconstruction | Heatmap)
- Score timeline plot

Usage:
    python evaluate_video.py --checkpoint results/video_S01_xxx/best_model.pth --data-dir ./data/IPAD
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import cv2

from models.video_autoencoder import VideoAutoencoder
from utils.video_dataset import IPADDataset, VideoDataset


def get_dataset_class(data_dir, category):
    """Determine which dataset class to use."""
    data_path = Path(data_dir) / category
    if (data_path / 'training' / 'frames').exists():
        return IPADDataset
    elif (data_path / 'train').exists():
        return VideoDataset
    else:
        raise FileNotFoundError(f"Could not find valid dataset at {data_path}")


def denormalize(tensor):
    """Convert tensor from [-1,1] to [0,255] numpy array."""
    tensor = tensor * 0.5 + 0.5
    tensor = torch.clamp(tensor, 0, 1)
    if tensor.dim() == 4:  # [T, C, H, W]
        array = tensor.permute(0, 2, 3, 1).cpu().numpy()
    else:  # [C, H, W]
        array = tensor.permute(1, 2, 0).cpu().numpy()
    return (array * 255).astype(np.uint8)


def create_heatmap(error_map, size=None):
    """Create colored heatmap from error values."""
    error_np = error_map.squeeze().cpu().numpy()
    
    # Normalize to 0-255
    error_norm = (error_np - error_np.min()) / (error_np.max() - error_np.min() + 1e-8)
    error_uint8 = (error_norm * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(error_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    if size:
        heatmap = cv2.resize(heatmap, size)
    
    return heatmap


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
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = checkpoint.get('args', {})
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Training loss: {checkpoint.get('train_loss', 0):.6f}")
    
    # Get parameters from saved args
    category = args.category or saved_args.get('category', 'S01')
    sequence_length = saved_args.get('sequence_length', 16)
    image_size = saved_args.get('image_size', 256)
    latent_dim = saved_args.get('latent_dim', 128)
    lstm_hidden_dim = saved_args.get('lstm_hidden_dim', 128)
    lstm_layers = saved_args.get('lstm_layers', 2)
    
    print(f"\nEvaluating on category: {category}")
    
    # Create model
    model = VideoAutoencoder(
        in_channels=3,
        latent_dim=latent_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_num_layers=lstm_layers
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load test dataset
    DatasetClass = get_dataset_class(args.data_dir, category)
    
    test_dataset = DatasetClass(
        root_dir=args.data_dir,
        category=category,
        split='test',
        sequence_length=sequence_length,
        stride=sequence_length,  # Non-overlapping for evaluation
        image_size=image_size
    )
    
    print(f"Test sequences: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Collect predictions
    all_scores = []
    all_labels = []
    all_frame_scores = []
    all_frame_labels = []
    
    print("\nComputing anomaly scores...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            sequences = batch['frames'].to(device)
            labels = batch['label'].numpy()
            
            # Get per-sequence error
            seq_errors = model.get_reconstruction_error(sequences, per_frame=False)
            all_scores.extend(seq_errors.cpu().numpy())
            all_labels.extend(labels)
            
            # Get per-frame errors if available
            if 'frame_labels' in batch and batch.get('frame_labels') is not None:
                frame_errors = model.get_reconstruction_error(sequences, per_frame=True)
                for i, frame_err in enumerate(frame_errors.cpu().numpy()):
                    all_frame_scores.extend(frame_err)
                    if batch['frame_labels'][i] is not None:
                        all_frame_labels.extend(batch['frame_labels'][i].numpy())
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    print("\n" + "="*50)
    
    # Sequence-level AUROC
    if len(np.unique(all_labels)) > 1:
        auroc = roc_auc_score(all_labels, all_scores)
        print(f"Sequence-level AUROC: {auroc:.4f}")
    else:
        auroc = 0.0
        print("Cannot compute AUROC - only one class present")
    
    # Frame-level AUROC if available
    if len(all_frame_scores) > 0 and len(all_frame_labels) > 0:
        all_frame_scores = np.array(all_frame_scores)
        all_frame_labels = np.array(all_frame_labels)
        if len(np.unique(all_frame_labels)) > 1:
            frame_auroc = roc_auc_score(all_frame_labels, all_frame_scores)
            print(f"Frame-level AUROC: {frame_auroc:.4f}")
    
    # Score statistics
    normal_scores = all_scores[all_labels == 0]
    anomaly_scores = all_scores[all_labels == 1]
    
    print("="*50)
    print("\nScore Statistics:")
    print(f"  Normal  - mean: {normal_scores.mean():.6f}, std: {normal_scores.std():.6f}")
    if len(anomaly_scores) > 0:
        print(f"  Anomaly - mean: {anomaly_scores.mean():.6f}, std: {anomaly_scores.std():.6f}")
        print(f"  Separation ratio: {anomaly_scores.mean() / normal_scores.mean():.2f}x")
    
    # Create output directory
    checkpoint_dir = Path(args.checkpoint).parent
    eval_dir = checkpoint_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    
    # Save ROC curve
    if len(np.unique(all_labels)) > 1:
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUROC = {auroc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Video Anomaly Detection\n{category}')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(eval_dir / 'roc_curve.png', dpi=150)
        plt.close()
        print(f"\nSaved ROC curve to {eval_dir / 'roc_curve.png'}")
    
    # Save score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(normal_scores, bins=30, alpha=0.7, label='Normal', color='green')
    if len(anomaly_scores) > 0:
        plt.hist(anomaly_scores, bins=30, alpha=0.7, label='Anomaly', color='red')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    plt.title(f'Score Distribution - {category}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(eval_dir / 'score_distribution.png', dpi=150)
    plt.close()
    print(f"Saved score distribution to {eval_dir / 'score_distribution.png'}")
    
    # Generate side-by-side visualization for a few sequences
    print("\nGenerating visualizations...")
    generate_visualizations(model, test_dataset, device, eval_dir, num_samples=4)
    
    # Save results text
    with open(eval_dir / 'results.txt', 'w') as f:
        f.write(f"Video Anomaly Detection Evaluation\n")
        f.write(f"Developed by Kuldeep Choksi\n")
        f.write(f"="*50 + "\n\n")
        f.write(f"Category: {category}\n")
        f.write(f"Sequence-level AUROC: {auroc:.4f}\n")
        f.write(f"Test sequences: {len(test_dataset)}\n")
        f.write(f"  Normal: {len(normal_scores)}\n")
        f.write(f"  Anomaly: {len(anomaly_scores)}\n\n")
        f.write(f"Score Statistics:\n")
        f.write(f"  Normal mean: {normal_scores.mean():.6f}\n")
        if len(anomaly_scores) > 0:
            f.write(f"  Anomaly mean: {anomaly_scores.mean():.6f}\n")
            f.write(f"  Separation: {anomaly_scores.mean() / normal_scores.mean():.2f}x\n")
    
    print(f"\nResults saved to: {eval_dir}")
    
    return auroc


def generate_visualizations(model, dataset, device, output_dir, num_samples=4):
    """Generate side-by-side visualization images."""
    
    # Get samples - mix of normal and anomaly if possible
    normal_indices = [i for i, s in enumerate(dataset.sequences) if s['label'] == 0]
    anomaly_indices = [i for i, s in enumerate(dataset.sequences) if s['label'] == 1]
    
    sample_indices = []
    if normal_indices:
        sample_indices.extend(normal_indices[:num_samples//2])
    if anomaly_indices:
        sample_indices.extend(anomaly_indices[:num_samples//2])
    
    if not sample_indices:
        sample_indices = list(range(min(num_samples, len(dataset))))
    
    for idx in sample_indices:
        sample = dataset[idx]
        frames = sample['frames'].unsqueeze(0).to(device)
        label = sample['label']
        label_name = 'ANOMALY' if label == 1 else 'NORMAL'
        
        with torch.no_grad():
            reconstruction = model(frames)
            error_maps = model.get_reconstruction_error(frames, per_pixel=True)
            seq_error = model.get_reconstruction_error(frames, per_frame=False)
        
        # Create visualization for middle frame
        T = frames.shape[1]
        mid_frame = T // 2
        
        orig = denormalize(frames[0, mid_frame])
        recon = denormalize(reconstruction[0, mid_frame])
        heatmap = create_heatmap(error_maps[0, mid_frame], size=(256, 256))
        
        # Combine side by side
        combined = np.hstack([orig, recon, heatmap])
        
        # Add text
        combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        cv2.putText(combined, 'Original', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, 'Reconstruction', (266, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, 'Error Heatmap', (522, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, f'{label_name} | Score: {seq_error.item():.4f}', (10, 250), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if label == 0 else (0, 0, 255), 2)
        combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        
        # Save
        plt.figure(figsize=(12, 4))
        plt.imshow(combined)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / f'visualization_{idx}_{label_name.lower()}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {len(sample_indices)} visualizations")


def generate_video_output(model, video_path, output_path, device, sequence_length=16):
    """
    Generate side-by-side annotated video output.
    
    Creates a video with three panels:
    - Original frame
    - Reconstruction
    - Error heatmap
    
    Plus a timeline showing anomaly score per frame.
    """
    from utils.video_dataset import VideoFileDataset
    
    dataset = VideoFileDataset(
        video_path=video_path,
        sequence_length=sequence_length,
        stride=1
    )
    
    if len(dataset) == 0:
        print("Video too short for analysis")
        return
    
    # Setup video writer
    frame_width = 256 * 3  # Side by side
    frame_height = 256 + 60  # Extra for score bar
    fps = dataset.fps
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    all_scores = []
    
    print(f"Processing {len(dataset)} sequences...")
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(loader):
            frames = batch['frames'].to(device)
            
            reconstruction = model(frames)
            error_maps = model.get_reconstruction_error(frames, per_pixel=True)
            frame_scores = model.get_reconstruction_error(frames, per_frame=True)
            
            # Process each frame in sequence
            for t in range(frames.shape[1]):
                orig = denormalize(frames[0, t])
                recon = denormalize(reconstruction[0, t])
                heatmap = create_heatmap(error_maps[0, t], size=(256, 256))
                score = frame_scores[0, t].item()
                
                all_scores.append(score)
                
                # Combine panels
                combined = np.hstack([orig, recon, heatmap])
                
                # Add score bar at bottom
                score_bar = np.zeros((60, frame_width, 3), dtype=np.uint8)
                
                # Draw score indicator
                score_norm = min(score / 0.01, 1.0)  # Normalize
                bar_width = int(score_norm * (frame_width - 20))
                color = (0, 255, 0) if score_norm < 0.5 else (0, 165, 255) if score_norm < 0.75 else (0, 0, 255)
                cv2.rectangle(score_bar, (10, 20), (10 + bar_width, 50), color, -1)
                cv2.rectangle(score_bar, (10, 20), (frame_width - 10, 50), (255, 255, 255), 2)
                
                # Add text
                cv2.putText(score_bar, f'Score: {score:.6f}', (10, 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Combine
                full_frame = np.vstack([combined, score_bar])
                
                # Write frame
                full_frame_bgr = cv2.cvtColor(full_frame, cv2.COLOR_RGB2BGR)
                out.write(full_frame_bgr)
    
    out.release()
    print(f"Saved annotated video to: {output_path}")
    
    # Save score timeline
    plt.figure(figsize=(12, 4))
    plt.plot(all_scores, 'b-', linewidth=0.5)
    plt.xlabel('Frame')
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Score Timeline')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    timeline_path = Path(output_path).parent / 'score_timeline.png'
    plt.savefig(timeline_path, dpi=150)
    plt.close()
    print(f"Saved score timeline to: {timeline_path}")
    
    return all_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate video anomaly detection model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='./data/IPAD',
                        help='Path to dataset')
    parser.add_argument('--category', type=str, default=None,
                        help='Dataset category (auto-detected from checkpoint if not provided)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for evaluation')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to single video file for inference')
    parser.add_argument('--output-video', type=str, default=None,
                        help='Path for output annotated video')
    
    args = parser.parse_args()
    
    print("="*60)
    print("VIDEO ANOMALY DETECTION EVALUATION")
    print("Developed by Kuldeep Choksi")
    print("="*60)
    
    if args.video:
        # Single video inference mode
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        saved_args = checkpoint.get('args', {})
        
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        model = VideoAutoencoder(
            in_channels=3,
            latent_dim=saved_args.get('latent_dim', 128),
            lstm_hidden_dim=saved_args.get('lstm_hidden_dim', 128),
            lstm_num_layers=saved_args.get('lstm_layers', 2)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        output_path = args.output_video or 'output_annotated.mp4'
        generate_video_output(model, args.video, output_path, device)
    else:
        # Dataset evaluation mode
        evaluate(args)