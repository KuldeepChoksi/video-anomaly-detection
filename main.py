"""
Anomaly Detection - Interactive Application
Developed by Kuldeep Choksi

A user-friendly interface for training, evaluating, and using
anomaly detection models on industrial images AND videos.

Supports:
- Image anomaly detection (MVTec AD, custom datasets)
- Video/temporal anomaly detection (IPAD, custom video datasets)

Run with: python main.py
Then open http://localhost:7860 in your browser.
"""

import gradio as gr
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import io
import os

from models import ConvAutoencoder
from utils import MVTecDataset


# ============================================================================
# GLOBAL STATE
# ============================================================================

class AppState:
    """Holds application state across UI interactions."""
    def __init__(self):
        self.model = None
        self.video_model = None
        self.device = self._get_device()
        self.checkpoint_path = None
        self.data_dir = "./data/original"
        self.category = "bottle"
        
    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

state = AppState()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_available_checkpoints(model_type="image"):
    """Scan results folder for available model checkpoints."""
    results_dir = Path("./results")
    if not results_dir.exists():
        return []
    
    checkpoints = []
    for folder in sorted(results_dir.iterdir(), reverse=True):
        if folder.is_dir():
            best_model = folder / "best_model.pth"
            if best_model.exists():
                # Filter by model type
                if model_type == "video" and "video_" in folder.name:
                    checkpoints.append(str(best_model))
                elif model_type == "image" and "video_" not in folder.name:
                    checkpoints.append(str(best_model))
                elif model_type == "all":
                    checkpoints.append(str(best_model))
    return checkpoints


def get_available_categories():
    """Scan data folder for available image categories."""
    categories = []
    data_dirs = [Path("./data"), Path("./data/original")]
    
    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        for folder in sorted(data_dir.iterdir()):
            if folder.is_dir() and (folder / "train").exists():
                cat_name = folder.name
                if cat_name not in categories:
                    categories.append(cat_name)
    
    return categories if categories else ["No datasets found - see Help tab"]


def get_available_video_categories():
    """Scan data folder for available video categories (IPAD format)."""
    categories = []
    
    # Check IPAD folder
    ipad_dir = Path("./data/IPAD")
    if ipad_dir.exists():
        for folder in sorted(ipad_dir.iterdir()):
            if folder.is_dir() and (folder / "training" / "frames").exists():
                categories.append(folder.name)
    
    # Check for custom video datasets in ./data
    data_dir = Path("./data")
    if data_dir.exists():
        for folder in sorted(data_dir.iterdir()):
            if folder.is_dir() and folder.name != "IPAD" and folder.name != "original":
                if (folder / "train").exists():
                    # Check if it has video structure
                    train_dir = folder / "train"
                    for sub in train_dir.iterdir():
                        if sub.is_dir():
                            # Check for video files or frame folders
                            has_videos = any(f.suffix in ['.mp4', '.avi', '.mov'] for f in sub.iterdir() if f.is_file())
                            has_frames = any(f.is_dir() for f in sub.iterdir())
                            if has_videos or has_frames:
                                if folder.name not in categories:
                                    categories.append(folder.name)
                                break
    
    return categories if categories else ["No video datasets found - see Help tab"]


def load_model_from_checkpoint(checkpoint_path):
    """Load a trained image model from checkpoint."""
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return None, "Error: Checkpoint file not found."
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=state.device, weights_only=False)
        args = checkpoint.get('args', {})
        latent_dim = args.get('latent_dim', 256)
        
        model = ConvAutoencoder(in_channels=3, latent_dim=latent_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(state.device)
        model.eval()
        
        state.model = model
        state.checkpoint_path = checkpoint_path
        state.category = args.get('category', 'bottle')
        
        epoch = checkpoint.get('epoch', 'unknown')
        train_loss = checkpoint.get('train_loss', 0)
        
        return model, f"""Model loaded successfully.

Details:
- Epoch: {epoch}
- Training Loss: {train_loss:.6f}
- Category: {state.category}
- Device: {state.device}

Model developed by Kuldeep Choksi"""
    
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


def load_video_model_from_checkpoint(checkpoint_path):
    """Load a trained video model from checkpoint."""
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return None, "Error: Checkpoint file not found."
    
    try:
        from models.video_autoencoder import VideoAutoencoder
        
        checkpoint = torch.load(checkpoint_path, map_location=state.device, weights_only=False)
        args = checkpoint.get('args', {})
        
        model = VideoAutoencoder(
            in_channels=3,
            latent_dim=args.get('latent_dim', 128),
            lstm_hidden_dim=args.get('lstm_hidden_dim', 128),
            lstm_num_layers=args.get('lstm_layers', 2)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(state.device)
        model.eval()
        
        state.video_model = model
        
        epoch = checkpoint.get('epoch', 'unknown')
        separation = checkpoint.get('separation', 0)
        category = args.get('category', 'unknown')
        
        return model, f"""Video model loaded successfully.

Details:
- Epoch: {epoch}
- Separation Ratio: {separation:.2f}x
- Category: {category}
- Device: {state.device}
- Sequence Length: {args.get('sequence_length', 16)} frames

Model developed by Kuldeep Choksi"""
    
    except Exception as e:
        return None, f"Error loading video model: {str(e)}"


def preprocess_image(image):
    """Preprocess uploaded image for model input."""
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    image = image.convert('RGB')
    tensor = transform(image).unsqueeze(0)
    return tensor


def denormalize(tensor):
    """Convert tensor from [-1,1] to [0,255] for display."""
    tensor = tensor * 0.5 + 0.5
    tensor = torch.clamp(tensor, 0, 1)
    array = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (array * 255).astype(np.uint8)


def create_error_heatmap(error_map):
    """Create a colored heatmap from error values."""
    error_np = error_map.squeeze().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(error_np, cmap='hot')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)


# ============================================================================
# IMAGE ANALYSIS CALLBACKS
# ============================================================================

def on_load_model(checkpoint_dropdown):
    """Callback when user clicks Load Model button."""
    if not checkpoint_dropdown:
        return "Please select a checkpoint from the dropdown first."
    
    model, message = load_model_from_checkpoint(checkpoint_dropdown)
    return message


def on_analyze_image(input_image):
    """Callback when user uploads an image for analysis."""
    if state.model is None:
        return None, None, "Please load a model first."
    
    if input_image is None:
        return None, None, "Please upload an image to analyze."
    
    try:
        input_tensor = preprocess_image(input_image).to(state.device)
        
        with torch.no_grad():
            reconstruction = state.model(input_tensor)
            error_map = state.model.get_reconstruction_error(input_tensor, per_pixel=True)
            error_score = state.model.get_reconstruction_error(input_tensor, per_pixel=False)
        
        recon_image = denormalize(reconstruction)
        heatmap_image = create_error_heatmap(error_map)
        
        score = error_score.item()
        threshold = 0.004
        is_anomaly = score > threshold
        status = "ANOMALY DETECTED" if is_anomaly else "NORMAL"
        confidence = min(abs(score - threshold) / threshold * 100, 100)
        
        report = f"""ANALYSIS COMPLETE - Kuldeep Choksi's Anomaly Detection System
{'='*60}

Status: {status}
Anomaly Score: {score:.6f}
Threshold: {threshold:.6f}
Confidence: {confidence:.1f}%

{'='*60}
INTERPRETATION

{"WARNING: Anomaly detected." if is_anomaly else "Image appears normal."}
"""
        
        return recon_image, heatmap_image, report
        
    except Exception as e:
        return None, None, f"Error during analysis: {str(e)}"


# ============================================================================
# VIDEO ANALYSIS CALLBACKS
# ============================================================================

def on_load_video_model(checkpoint_dropdown):
    """Load a video model."""
    if not checkpoint_dropdown:
        return "Please select a video model checkpoint first."
    
    model, message = load_video_model_from_checkpoint(checkpoint_dropdown)
    return message


def on_analyze_video(video_path):
    """Analyze uploaded video for anomalies."""
    if state.video_model is None:
        return None, "Please load a video model first."
    
    if video_path is None:
        return None, "Please upload a video file."
    
    try:
        from utils.video_dataset import VideoFileDataset
        from torch.utils.data import DataLoader
        import cv2
        
        # Create dataset from uploaded video
        dataset = VideoFileDataset(
            video_path=video_path,
            sequence_length=16,
            stride=8
        )
        
        if len(dataset) == 0:
            return None, "Video too short for analysis (need at least 16 frames)."
        
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        all_scores = []
        frame_results = []
        
        with torch.no_grad():
            for batch in loader:
                frames = batch['frames'].to(state.device)
                frame_scores = state.video_model.get_reconstruction_error(frames, per_frame=True)
                all_scores.extend(frame_scores[0].cpu().numpy())
        
        # Create score timeline plot
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(all_scores, 'b-', linewidth=1)
        ax.axhline(y=np.mean(all_scores) + 2*np.std(all_scores), color='r', linestyle='--', label='Threshold')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Anomaly Score')
        ax.set_title('Video Anomaly Score Timeline - Kuldeep Choksi')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close()
        
        timeline_image = Image.open(buf)
        
        # Generate report
        mean_score = np.mean(all_scores)
        max_score = np.max(all_scores)
        threshold = mean_score + 2*np.std(all_scores)
        anomaly_frames = np.where(np.array(all_scores) > threshold)[0]
        
        report = f"""VIDEO ANALYSIS COMPLETE - Kuldeep Choksi's Anomaly Detection System
{'='*60}

Video Statistics:
- Total Frames Analyzed: {len(all_scores)}
- Mean Anomaly Score: {mean_score:.6f}
- Max Anomaly Score: {max_score:.6f}
- Threshold (mean + 2*std): {threshold:.6f}

Anomaly Detection:
- Frames Above Threshold: {len(anomaly_frames)}
- Anomaly Percentage: {100*len(anomaly_frames)/len(all_scores):.1f}%

{'='*60}
"""
        if len(anomaly_frames) > 0:
            report += f"\nPotential anomaly frames: {anomaly_frames[:20].tolist()}"
            if len(anomaly_frames) > 20:
                report += f"\n... and {len(anomaly_frames)-20} more"
        else:
            report += "\nNo significant anomalies detected."
        
        return timeline_image, report
        
    except Exception as e:
        return None, f"Error analyzing video: {str(e)}"


# ============================================================================
# TRAINING CALLBACKS
# ============================================================================

def on_start_training(category, epochs, batch_size, learning_rate, loss_type):
    """Start image model training."""
    import subprocess
    import sys
    
    if not category or category == "No datasets found - see Help tab":
        yield "Error: No dataset found."
        return
    
    if (Path("./data") / category / "train").exists():
        data_dir = "./data"
    elif (Path("./data/original") / category / "train").exists():
        data_dir = "./data/original"
    else:
        yield f"Error: Could not find dataset for '{category}'."
        return
    
    cmd = [
        sys.executable, "train.py",
        "--category", category,
        "--data-dir", data_dir,
        "--epochs", str(int(epochs)),
        "--batch-size", str(int(batch_size)),
        "--lr", str(learning_rate),
        "--loss", loss_type
    ]
    
    header = f"""TRAINING STARTED - Kuldeep Choksi's Anomaly Detection System
{'='*60}

Configuration:
- Category: {category}
- Epochs: {int(epochs)}
- Batch Size: {int(batch_size)}
- Loss: {loss_type}

{'='*60}
"""
    
    yield header + "Initializing...\n"
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        output_lines = []
        
        for line in process.stdout:
            line = line.strip()
            if line:
                output_lines.append(line)
                display_lines = output_lines[-40:]
                yield header + '\n'.join(display_lines)
        
        process.wait()
        yield header + '\n'.join(output_lines[-40:]) + "\n\nTRAINING COMPLETE"
    except Exception as e:
        yield f"Error during training: {str(e)}"


def on_start_video_training(category, epochs, batch_size, learning_rate):
    """Start video model training."""
    import subprocess
    import sys
    
    if not category or category == "No video datasets found - see Help tab":
        yield "Error: No video dataset found."
        return
    
    # Determine data directory
    if (Path("./data/IPAD") / category / "training" / "frames").exists():
        data_dir = "./data/IPAD"
    elif (Path("./data") / category / "train").exists():
        data_dir = "./data"
    else:
        yield f"Error: Could not find video dataset for '{category}'."
        return
    
    cmd = [
        sys.executable, "train_video.py",
        "--category", category,
        "--data-dir", data_dir,
        "--epochs", str(int(epochs)),
        "--batch-size", str(int(batch_size)),
        "--lr", str(learning_rate)
    ]
    
    header = f"""VIDEO TRAINING STARTED - Kuldeep Choksi's Anomaly Detection System
{'='*60}

Configuration:
- Category: {category}
- Data Directory: {data_dir}
- Epochs: {int(epochs)}
- Batch Size: {int(batch_size)}
- Learning Rate: {learning_rate}

*** Saving based on SEPARATION RATIO (not loss) ***

{'='*60}
"""
    
    yield header + "Initializing...\n"
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        output_lines = []
        
        for line in process.stdout:
            line = line.strip()
            if line:
                output_lines.append(line)
                display_lines = output_lines[-40:]
                yield header + '\n'.join(display_lines)
        
        process.wait()
        yield header + '\n'.join(output_lines[-40:]) + "\n\nTRAINING COMPLETE"
    except Exception as e:
        yield f"Error during training: {str(e)}"


# ============================================================================
# RESULTS CALLBACKS
# ============================================================================

def on_view_results(checkpoint_path):
    """Load and display evaluation results."""
    if not checkpoint_path:
        return None, None, None, "Please select a checkpoint first."
    
    checkpoint_dir = Path(checkpoint_path).parent
    eval_dir = checkpoint_dir / "evaluation"
    
    if not eval_dir.exists():
        return None, None, None, f"No evaluation results found. Run evaluation first."
    
    roc_path = eval_dir / "roc_curve.png"
    dist_path = eval_dir / "score_distribution.png"
    recon_path = eval_dir / "reconstructions.png"
    results_path = eval_dir / "results.txt"
    
    roc_img = Image.open(roc_path) if roc_path.exists() else None
    dist_img = Image.open(dist_path) if dist_path.exists() else None
    recon_img = Image.open(recon_path) if recon_path.exists() else None
    
    if results_path.exists():
        with open(results_path, 'r') as f:
            results_text = f.read()
    else:
        results_text = "No results.txt found."
    
    return roc_img, dist_img, recon_img, results_text


def on_run_evaluation(checkpoint_path):
    """Run evaluation on selected model."""
    import subprocess
    import sys
    
    if not checkpoint_path:
        return "Please select a checkpoint first.", None, None, None, ""
    
    # Determine if video or image model
    is_video = "video_" in checkpoint_path
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    category = checkpoint.get('args', {}).get('category', 'bottle')
    
    # Determine data directory
    if is_video:
        if (Path("./data/IPAD") / category / "training").exists():
            data_dir = "./data/IPAD"
        else:
            data_dir = "./data"
        eval_script = "evaluate_video.py"
    else:
        if (Path("./data") / category / "train").exists():
            data_dir = "./data"
        elif (Path("./data/original") / category / "train").exists():
            data_dir = "./data/original"
        else:
            return f"Error: Dataset not found for {category}", None, None, None, ""
        eval_script = "evaluate.py"
    
    cmd = [sys.executable, eval_script, "--checkpoint", checkpoint_path, "--data-dir", data_dir, "--category", category]
    
    status = f"Running evaluation on {category}...\n\n"
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output_lines = []
        for line in process.stdout:
            output_lines.append(line.strip())
        process.wait()
        
        status += '\n'.join(output_lines) + "\n\nEVALUATION COMPLETE"
        
        roc_img, dist_img, recon_img, summary = on_view_results(checkpoint_path)
        return status, roc_img, dist_img, recon_img, summary
        
    except Exception as e:
        return f"Error: {str(e)}", None, None, None, ""


# ============================================================================
# BUILD THE UI
# ============================================================================

def create_ui():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="Anomaly Detection System") as app:
        
        gr.Markdown("""
        # Industrial Anomaly Detection System
        **Developed by Kuldeep Choksi**
        
        Detect manufacturing defects in **images** and **videos** using deep learning.
        
        ---
        """)
        
        with gr.Tabs():
            
            # =================================================================
            # TAB 1: ANALYZE IMAGES
            # =================================================================
            with gr.Tab("Analyze Images"):
                gr.Markdown("## Image Anomaly Detection\n*Developed by Kuldeep Choksi*")
                
                gr.Markdown("### Step 1: Load a Model")
                with gr.Row():
                    checkpoint_dropdown = gr.Dropdown(
                        choices=get_available_checkpoints("image"),
                        label="Select Image Model"
                    )
                    refresh_btn = gr.Button("Refresh", variant="secondary")
                    load_btn = gr.Button("Load Model", variant="primary")
                
                model_status = gr.Textbox(label="Model Status", lines=6, value="No model loaded.")
                
                gr.Markdown("### Step 2: Upload and Analyze")
                with gr.Row():
                    input_image = gr.Image(label="Upload Image", type="pil")
                    reconstruction_output = gr.Image(label="Reconstruction")
                    heatmap_output = gr.Image(label="Error Heatmap")
                
                analyze_btn = gr.Button("Analyze Image", variant="primary", size="lg")
                analysis_report = gr.Textbox(label="Analysis Report", lines=15)
                
                refresh_btn.click(fn=lambda: gr.Dropdown(choices=get_available_checkpoints("image")), outputs=checkpoint_dropdown)
                load_btn.click(fn=on_load_model, inputs=checkpoint_dropdown, outputs=model_status)
                analyze_btn.click(fn=on_analyze_image, inputs=input_image, outputs=[reconstruction_output, heatmap_output, analysis_report])
            
            # =================================================================
            # TAB 2: ANALYZE VIDEO
            # =================================================================
            with gr.Tab("Analyze Video"):
                gr.Markdown("## Video Anomaly Detection\n*Developed by Kuldeep Choksi*")
                
                gr.Markdown("### Step 1: Load a Video Model")
                with gr.Row():
                    video_checkpoint_dropdown = gr.Dropdown(
                        choices=get_available_checkpoints("video"),
                        label="Select Video Model"
                    )
                    video_refresh_btn = gr.Button("Refresh", variant="secondary")
                    video_load_btn = gr.Button("Load Model", variant="primary")
                
                video_model_status = gr.Textbox(label="Model Status", lines=6, value="No video model loaded.")
                
                gr.Markdown("### Step 2: Upload and Analyze Video")
                video_input = gr.Video(label="Upload Video")
                video_analyze_btn = gr.Button("Analyze Video", variant="primary", size="lg")
                
                timeline_output = gr.Image(label="Anomaly Score Timeline")
                video_report = gr.Textbox(label="Analysis Report", lines=15)
                
                video_refresh_btn.click(fn=lambda: gr.Dropdown(choices=get_available_checkpoints("video")), outputs=video_checkpoint_dropdown)
                video_load_btn.click(fn=on_load_video_model, inputs=video_checkpoint_dropdown, outputs=video_model_status)
                video_analyze_btn.click(fn=on_analyze_video, inputs=video_input, outputs=[timeline_output, video_report])
            
            # =================================================================
            # TAB 3: TRAIN IMAGE MODEL
            # =================================================================
            with gr.Tab("Train Image Model"):
                gr.Markdown("## Train Image Anomaly Detection Model\n*Developed by Kuldeep Choksi*")
                
                with gr.Row():
                    with gr.Column():
                        category_dropdown = gr.Dropdown(choices=get_available_categories(), value="bottle", label="Dataset Category")
                        epochs_slider = gr.Slider(minimum=10, maximum=200, value=50, step=10, label="Epochs")
                        batch_slider = gr.Slider(minimum=4, maximum=64, value=16, step=4, label="Batch Size")
                    with gr.Column():
                        lr_dropdown = gr.Dropdown(choices=[0.0001, 0.0005, 0.001, 0.005], value=0.001, label="Learning Rate")
                        loss_dropdown = gr.Dropdown(choices=["mse", "ssim", "combined"], value="mse", label="Loss Function")
                
                train_btn = gr.Button("Start Training", variant="primary", size="lg")
                training_output = gr.Textbox(label="Training Status", lines=25)
                
                train_btn.click(fn=on_start_training, inputs=[category_dropdown, epochs_slider, batch_slider, lr_dropdown, loss_dropdown], outputs=training_output)
            
            # =================================================================
            # TAB 4: TRAIN VIDEO MODEL
            # =================================================================
            with gr.Tab("Train Video Model"):
                gr.Markdown("## Train Video Anomaly Detection Model\n*Developed by Kuldeep Choksi*")
                
                gr.Markdown("""
                Train a ConvLSTM model on video sequences. The model learns temporal patterns
                of normal operation and detects anomalies as deviations from these patterns.
                
                **Note:** Training saves based on separation ratio (not loss) to ensure best anomaly detection.
                """)
                
                with gr.Row():
                    with gr.Column():
                        video_category = gr.Dropdown(choices=get_available_video_categories(), label="Video Dataset")
                        video_epochs = gr.Slider(minimum=5, maximum=50, value=10, step=5, label="Epochs")
                    with gr.Column():
                        video_batch = gr.Slider(minimum=2, maximum=16, value=4, step=2, label="Batch Size")
                        video_lr = gr.Dropdown(choices=[0.0001, 0.0005, 0.001], value=0.0001, label="Learning Rate")
                
                video_train_btn = gr.Button("Start Video Training", variant="primary", size="lg")
                video_training_output = gr.Textbox(label="Training Status", lines=25)
                
                video_train_btn.click(fn=on_start_video_training, inputs=[video_category, video_epochs, video_batch, video_lr], outputs=video_training_output)
            
            # =================================================================
            # TAB 5: VIEW RESULTS
            # =================================================================
            with gr.Tab("View Results"):
                gr.Markdown("## Evaluation Results\n*Developed by Kuldeep Choksi*")
                
                with gr.Row():
                    results_checkpoint = gr.Dropdown(choices=get_available_checkpoints("all"), label="Select Model")
                    results_refresh_btn = gr.Button("Refresh", variant="secondary")
                
                with gr.Row():
                    run_eval_btn = gr.Button("Run Evaluation", variant="primary")
                    view_results_btn = gr.Button("View Existing Results", variant="secondary")
                
                eval_status = gr.Textbox(label="Evaluation Status", lines=15)
                results_summary = gr.Textbox(label="Results Summary", lines=8)
                
                with gr.Row():
                    roc_image = gr.Image(label="ROC Curve")
                    dist_image = gr.Image(label="Score Distribution")
                
                recon_image = gr.Image(label="Sample Visualizations")
                
                results_refresh_btn.click(fn=lambda: gr.Dropdown(choices=get_available_checkpoints("all")), outputs=results_checkpoint)
                run_eval_btn.click(fn=on_run_evaluation, inputs=results_checkpoint, outputs=[eval_status, roc_image, dist_image, recon_image, results_summary])
                view_results_btn.click(fn=on_view_results, inputs=results_checkpoint, outputs=[roc_image, dist_image, recon_image, results_summary])
            
            # =================================================================
            # TAB 6: HELP
            # =================================================================
            with gr.Tab("Help"):
                gr.Markdown("""
                ## Help and Documentation
                *Documentation by Kuldeep Choksi*
                
                ---
                
                ### Image Anomaly Detection
                
                Train on normal images, detect defects as reconstruction failures.
                
                **Folder structure for custom image datasets:**
                ```
                data/your_category/
                    train/good/          (normal images)
                    test/good/           (normal test images)
                    test/defect_type/    (defective images)
                ```
                
                ---
                
                ### Video Anomaly Detection
                
                Train on normal video sequences, detect temporal anomalies.
                
                **Folder structure for custom video datasets:**
                ```
                data/your_category/
                    train/normal/
                        video_001.mp4    (or folder of frames)
                        video_002.mp4
                    test/normal/
                        test_001.mp4
                    test/anomaly/
                        anomaly_001.mp4
                ```
                
                **IPAD Dataset:** Pre-formatted industrial video data in `data/IPAD/`
                
                ---
                
                ### Interpreting Results
                
                - **AUROC**: 0.5 = random, 0.7-0.8 = good, 0.9+ = excellent
                - **Separation Ratio**: Higher = better anomaly detection
                - **Error Heatmap**: Red/yellow = high error = potential anomaly
                
                ---
                
                ### About
                
                Developed by **Kuldeep Choksi** as a computer vision portfolio project.
                
                **Results:**
                - Image: 0.89 AUROC on MVTec bottles
                - Video: 0.85 AUROC on IPAD R01
                
                [GitHub](https://github.com/KuldeepChoksi)
                """)
        
        gr.Markdown("---\n*Anomaly Detection System | Developed by Kuldeep Choksi | Built with PyTorch and Gradio*")
    
    return app


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("ANOMALY DETECTION SYSTEM")
    print("Developed by Kuldeep Choksi")
    print("="*60)
    print()
    print("Starting application...")
    print("Open: http://localhost:7860")
    print("Press Ctrl+C to stop")
    print("="*60)
    
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)