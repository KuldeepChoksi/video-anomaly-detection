"""
Video Dataset Loader for Temporal Anomaly Detection.
Developed by Kuldeep Choksi

Supports:
    - Video files (mp4, avi, mov)
    - Image sequence folders
    - IPAD dataset format
    - Custom video datasets

The loader creates sliding window sequences of frames for temporal modeling.
"""

import os
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List, Tuple, Optional


class IPADDataset(Dataset):
    """
    Dataset loader specifically for IPAD format.
    
    IPAD structure:
        category/
            training/frames/01/, 02/, ...  (frame folders)
            testing/frames/01/, 02/, ...
            test_label/001.npy, 002.npy, ... (per-frame labels)
    
    Args:
        root_dir: Path to IPAD dataset
        category: Which device (e.g., 'S01', 'R01')
        split: 'train' or 'test'
        sequence_length: Number of frames per sequence
        stride: Step between sequences
        image_size: Resize frames to this size
    """
    
    def __init__(
        self,
        root_dir: str,
        category: str,
        split: str = 'train',
        sequence_length: int = 16,
        stride: int = 4,
        transform=None,
        image_size: int = 256
    ):
        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split
        self.sequence_length = sequence_length
        self.stride = stride
        self.image_size = image_size
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
        
        self.sequences = []
        self._load_dataset()
    
    def _load_dataset(self):
        """Load IPAD dataset structure."""
        
        if self.split == 'train':
            frames_dir = self.root_dir / self.category / 'training' / 'frames'
            labels_dir = None
        else:
            frames_dir = self.root_dir / self.category / 'testing' / 'frames'
            labels_dir = self.root_dir / self.category / 'test_label'
        
        if not frames_dir.exists():
            raise FileNotFoundError(f"Dataset not found at {frames_dir}")
        
        # Each numbered folder is a video
        for video_folder in sorted(frames_dir.iterdir()):
            if not video_folder.is_dir():
                continue
            
            video_id = video_folder.name
            
            # Get all frame paths
            frame_files = sorted([
                f for f in video_folder.iterdir()
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
            ])
            
            if len(frame_files) < self.sequence_length:
                continue
            
            # Load labels if testing
            frame_labels = None
            if labels_dir:
                # Try different naming conventions
                label_file = labels_dir / f"{int(video_id):03d}.npy"
                if not label_file.exists():
                    label_file = labels_dir / f"{video_id}.npy"
                
                if label_file.exists():
                    frame_labels = np.load(label_file)
            
            # Create sequences
            for start_idx in range(0, len(frame_files) - self.sequence_length + 1, self.stride):
                end_idx = start_idx + self.sequence_length
                
                # Determine label for this sequence
                if frame_labels is not None:
                    # Sequence is anomaly if ANY frame in it is anomaly
                    seq_labels = frame_labels[start_idx:end_idx]
                    is_anomaly = 1 if np.any(seq_labels == 1) else 0
                else:
                    is_anomaly = 0  # Training data is all normal
                
                self.sequences.append({
                    'frame_paths': [str(f) for f in frame_files[start_idx:end_idx]],
                    'label': is_anomaly,
                    'video_id': video_id,
                    'start_frame': start_idx,
                    'frame_labels': frame_labels[start_idx:end_idx] if frame_labels is not None else None
                })
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        
        frames = []
        for frame_path in seq_info['frame_paths']:
            img = Image.open(frame_path).convert('RGB')
            frames.append(self.transform(img))
        
        frames_tensor = torch.stack(frames, dim=0)
        
        return {
            'frames': frames_tensor,
            'label': seq_info['label'],
            'video_id': seq_info['video_id'],
            'start_frame': seq_info['start_frame'],
            'label_name': 'anomaly' if seq_info['label'] == 1 else 'normal'
        }


class VideoDataset(Dataset):
    """
    Dataset for loading video sequences for temporal anomaly detection.
    
    Creates sliding window sequences of N consecutive frames.
    The model learns temporal patterns from normal sequences.
    
    Args:
        root_dir: Path to dataset
        category: Which device/category (e.g., 'automatic_lifter')
        split: 'train' or 'test'
        sequence_length: Number of frames per sequence (default: 16)
        stride: Step between sequences (default: 1 for dense sampling)
        transform: Optional torchvision transforms
        image_size: Resize frames to this size
    """
    
    def __init__(
        self,
        root_dir: str,
        category: str,
        split: str = 'train',
        sequence_length: int = 16,
        stride: int = 4,
        transform=None,
        image_size: int = 256
    ):
        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split
        self.sequence_length = sequence_length
        self.stride = stride
        self.image_size = image_size
        
        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
        
        # Storage for sequences
        self.sequences = []  # List of (video_path, start_frame, label)
        self.frame_cache = {}  # Cache loaded frames
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Scan directory and build list of all sequences."""
        
        split_dir = self.root_dir / self.category / self.split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Dataset not found at {split_dir}")
        
        # Find all video files or frame folders
        for label_folder in sorted(split_dir.iterdir()):
            if not label_folder.is_dir():
                continue
            
            label_name = label_folder.name
            is_anomaly = 0 if label_name in ['good', 'normal', 'train'] else 1
            
            # Check for video files
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            for video_file in sorted(label_folder.iterdir()):
                if video_file.suffix.lower() in video_extensions:
                    self._add_sequences_from_video(video_file, is_anomaly, label_name)
                elif video_file.is_dir():
                    # Frame folder
                    self._add_sequences_from_frames(video_file, is_anomaly, label_name)
    
    def _add_sequences_from_video(self, video_path: Path, label: int, label_name: str):
        """Extract sequence indices from a video file."""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if total_frames < self.sequence_length:
            return
        
        # Create sequences with sliding window
        for start_idx in range(0, total_frames - self.sequence_length + 1, self.stride):
            self.sequences.append({
                'source': str(video_path),
                'source_type': 'video',
                'start_frame': start_idx,
                'label': label,
                'label_name': label_name
            })
    
    def _add_sequences_from_frames(self, frame_dir: Path, label: int, label_name: str):
        """Extract sequence indices from a folder of frame images."""
        frame_files = sorted([
            f for f in frame_dir.iterdir() 
            if f.suffix.lower() in ['.png', '.jpg', '.jpeg']
        ])
        
        total_frames = len(frame_files)
        if total_frames < self.sequence_length:
            return
        
        # Store frame paths
        frame_paths = [str(f) for f in frame_files]
        
        # Create sequences with sliding window
        for start_idx in range(0, total_frames - self.sequence_length + 1, self.stride):
            self.sequences.append({
                'source': frame_paths,
                'source_type': 'frames',
                'start_frame': start_idx,
                'label': label,
                'label_name': label_name
            })
    
    def _load_frames_from_video(self, video_path: str, start_idx: int) -> List[Image.Image]:
        """Load sequence of frames from video file."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        
        frames = []
        for _ in range(self.sequence_length):
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        
        cap.release()
        return frames
    
    def _load_frames_from_folder(self, frame_paths: List[str], start_idx: int) -> List[Image.Image]:
        """Load sequence of frames from image files."""
        frames = []
        for i in range(start_idx, start_idx + self.sequence_length):
            if i < len(frame_paths):
                frames.append(Image.open(frame_paths[i]).convert('RGB'))
        return frames
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Returns:
            frames: Tensor of shape [T, C, H, W] where T is sequence_length
            label: 0 for normal, 1 for anomaly
        """
        seq_info = self.sequences[idx]
        
        # Load frames
        if seq_info['source_type'] == 'video':
            frames = self._load_frames_from_video(seq_info['source'], seq_info['start_frame'])
        else:
            frames = self._load_frames_from_folder(seq_info['source'], seq_info['start_frame'])
        
        # Apply transforms to each frame
        transformed_frames = []
        for frame in frames:
            transformed_frames.append(self.transform(frame))
        
        # Stack to [T, C, H, W]
        frames_tensor = torch.stack(transformed_frames, dim=0)
        
        return {
            'frames': frames_tensor,
            'label': seq_info['label'],
            'label_name': seq_info['label_name'],
            'source': seq_info['source'] if isinstance(seq_info['source'], str) else seq_info['source'][0],
            'start_frame': seq_info['start_frame']
        }


class VideoFileDataset(Dataset):
    """
    Simple dataset for processing a single video file.
    Used for inference on user-uploaded videos.
    
    Args:
        video_path: Path to video file
        sequence_length: Number of frames per sequence
        stride: Step between sequences
        image_size: Resize frames to this size
    """
    
    def __init__(
        self,
        video_path: str,
        sequence_length: int = 16,
        stride: int = 1,
        image_size: int = 256
    ):
        self.video_path = video_path
        self.sequence_length = sequence_length
        self.stride = stride
        self.image_size = image_size
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Calculate number of sequences
        self.num_sequences = max(0, (self.total_frames - sequence_length) // stride + 1)
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start_frame = idx * self.stride
        
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        original_frames = []
        
        for _ in range(self.sequence_length):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frames.append(frame_rgb)
            pil_frame = Image.fromarray(frame_rgb)
            frames.append(self.transform(pil_frame))
        
        cap.release()
        
        frames_tensor = torch.stack(frames, dim=0)
        
        return {
            'frames': frames_tensor,
            'start_frame': start_frame,
            'original_frames': original_frames
        }


def get_video_dataloaders(
    root_dir: str,
    category: str,
    sequence_length: int = 16,
    stride: int = 4,
    batch_size: int = 8,
    image_size: int = 256,
    num_workers: int = 4
):
    """
    Convenience function to get train and test dataloaders for video.
    
    Returns:
        train_loader: DataLoader with only normal sequences
        test_loader: DataLoader with both normal and anomalous sequences
    """
    
    train_dataset = VideoDataset(
        root_dir=root_dir,
        category=category,
        split='train',
        sequence_length=sequence_length,
        stride=stride,
        image_size=image_size
    )
    
    test_dataset = VideoDataset(
        root_dir=root_dir,
        category=category,
        split='test',
        sequence_length=sequence_length,
        stride=stride,
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


# Quick test
if __name__ == '__main__':
    print("Video Dataset Loader")
    print("Developed by Kuldeep Choksi")
    print()
    print("Supports:")
    print("  - Video files (mp4, avi, mov)")
    print("  - Image sequence folders")
    print("  - IPAD dataset format")
    print("  - Custom video datasets")