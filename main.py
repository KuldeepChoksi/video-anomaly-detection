"""
Anomaly Detection - Interactive Application
Developed by Kuldeep Choksi

A user-friendly interface for training, evaluating, and using
anomaly detection models on industrial images.

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

def get_available_checkpoints():
    """Scan results folder for available model checkpoints."""
    results_dir = Path("./results")
    if not results_dir.exists():
        return []
    
    checkpoints = []
    for folder in sorted(results_dir.iterdir(), reverse=True):
        if folder.is_dir():
            best_model = folder / "best_model.pth"
            if best_model.exists():
                checkpoints.append(str(best_model))
    return checkpoints


def get_available_categories():
    """Scan data folder for available categories (including custom ones)."""
    categories = []
    
    # Check both ./data and ./data/original
    data_dirs = [Path("./data"), Path("./data/original")]
    
    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        
        for folder in sorted(data_dir.iterdir()):
            if folder.is_dir() and (folder / "train").exists():
                # Valid category folder found
                cat_name = folder.name
                if cat_name not in categories:
                    categories.append(cat_name)
    
    return categories if categories else ["No datasets found - see Help tab"]


def load_model_from_checkpoint(checkpoint_path):
    """Load a trained model from checkpoint."""
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
        
        return         model, f"Model loaded successfully.\n\nDetails:\n- Epoch: {epoch}\n- Training Loss: {train_loss:.6f}\n- Category: {state.category}\n- Device: {state.device}\n\nModel developed by Kuldeep Choksi"
    
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


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
# UI CALLBACK FUNCTIONS
# ============================================================================

def on_load_model(checkpoint_dropdown):
    """Callback when user clicks Load Model button."""
    if not checkpoint_dropdown:
        return "Please select a checkpoint from the dropdown first.\n\nIf no checkpoints appear, you need to train a model first. Go to the Training tab."
    
    model, message = load_model_from_checkpoint(checkpoint_dropdown)
    return message


def on_analyze_image(input_image):
    """Callback when user uploads an image for analysis."""
    if state.model is None:
        return None, None, "Please load a model first.\n\nStep 1: Go to 'Model Setup' section above\nStep 2: Select a checkpoint from the dropdown\nStep 3: Click 'Load Model'\nStep 4: Return here and upload an image"
    
    if input_image is None:
        return None, None, "Please upload an image to analyze.\n\nSupported formats: PNG, JPG, JPEG\nRecommended: Use images similar to training data (e.g., bottle images from MVTec)"
    
    try:
        # Preprocess
        input_tensor = preprocess_image(input_image).to(state.device)
        
        # Get reconstruction and error
        with torch.no_grad():
            reconstruction = state.model(input_tensor)
            error_map = state.model.get_reconstruction_error(input_tensor, per_pixel=True)
            error_score = state.model.get_reconstruction_error(input_tensor, per_pixel=False)
        
        # Convert to displayable images
        recon_image = denormalize(reconstruction)
        heatmap_image = create_error_heatmap(error_map)
        
        # Generate analysis report
        score = error_score.item()
        
        # Determine anomaly status (threshold based on typical values)
        threshold = 0.004  # Adjust based on your model's performance
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

The anomaly score represents reconstruction error.
- Lower scores indicate the image matches learned normal patterns
- Higher scores indicate deviations from normal patterns

{"WARNING: The image shows characteristics that differ from normal samples. This may indicate a defect or anomaly." if is_anomaly else "The image appears to match normal patterns learned during training."}

{'='*60}
NEXT STEPS

{"- Inspect the error heatmap to locate the anomaly region\n- Red/yellow areas indicate where reconstruction failed\n- Compare with the original to identify the defect" if is_anomaly else "- The model successfully reconstructed this image\n- Low error across the image indicates normal patterns\n- You may want to test with known defective images to validate"}
"""
        
        return recon_image, heatmap_image, report
        
    except Exception as e:
        return None, None, f"Error during analysis: {str(e)}\n\nPlease ensure your image is valid and try again."


def on_start_training(category, epochs, batch_size, learning_rate, loss_type):
    """Callback when user starts training - yields live progress updates."""
    import subprocess
    import sys
    
    # Validate inputs
    if not category or category == "No datasets found - see Help tab":
        yield "Error: No dataset found.\n\nPlease add your data following the folder structure in the Help tab."
        return
    
    # Determine correct data directory
    # Check if category exists in ./data or ./data/original
    if (Path("./data") / category / "train").exists():
        data_dir = "./data"
    elif (Path("./data/original") / category / "train").exists():
        data_dir = "./data/original"
    else:
        yield f"Error: Could not find dataset for '{category}'.\n\nMake sure the folder structure is correct:\n  data/{category}/train/good/\n\nSee Help tab for details."
        return
    
    # Build command
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
- Data Directory: {data_dir}
- Epochs: {int(epochs)}
- Batch Size: {int(batch_size)}
- Learning Rate: {learning_rate}
- Loss Function: {loss_type}

{'='*60}
LIVE TRAINING OUTPUT:
{'='*60}

"""
    
    yield header + "Initializing...\n"
    
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1
        )
        
        output_lines = []
        
        for line in process.stdout:
            line = line.strip()
            if line:
                output_lines.append(line)
                # Keep last 40 lines for display
                display_lines = output_lines[-40:]
                yield header + '\n'.join(display_lines)
        
        process.wait()
        
        # Final output
        final_output = '\n'.join(output_lines[-40:])
        
        yield f"""{header}{final_output}

{'='*60}
TRAINING COMPLETE
{'='*60}

Your model has been saved to the results/ folder.

NEXT STEPS:
1. Go to the 'Analyze Images' tab
2. Click 'Refresh List' next to the checkpoint dropdown
3. Select your new model (most recent timestamp)
4. Upload images to analyze for anomalies
"""
    except Exception as e:
        yield f"Error during training: {str(e)}"


def on_refresh_checkpoints():
    """Refresh the checkpoint dropdown."""
    return gr.Dropdown(choices=get_available_checkpoints())


def on_view_results(checkpoint_path):
    """Load and display evaluation results for a checkpoint."""
    if not checkpoint_path:
        return None, None, None, "Please select a checkpoint first."
    
    checkpoint_dir = Path(checkpoint_path).parent
    eval_dir = checkpoint_dir / "evaluation"
    
    if not eval_dir.exists():
        return None, None, None, f"""No evaluation results found for this model.

Click 'Run Evaluation' to generate results automatically.

Or run from terminal:
   python evaluate.py --checkpoint {checkpoint_path} --data-dir {state.data_dir}
"""
    
    # Load images
    roc_path = eval_dir / "roc_curve.png"
    dist_path = eval_dir / "score_distribution.png"
    recon_path = eval_dir / "reconstructions.png"
    results_path = eval_dir / "results.txt"
    
    roc_img = Image.open(roc_path) if roc_path.exists() else None
    dist_img = Image.open(dist_path) if dist_path.exists() else None
    recon_img = Image.open(recon_path) if recon_path.exists() else None
    
    # Load text results
    if results_path.exists():
        with open(results_path, 'r') as f:
            results_text = f.read()
    else:
        results_text = "No results.txt found."
    
    summary = f"""EVALUATION RESULTS - Kuldeep Choksi's Anomaly Detection System
{'='*60}

{results_text}

{'='*60}
Files saved at: {eval_dir}
"""
    
    return roc_img, dist_img, recon_img, summary


def on_run_evaluation(checkpoint_path):
    """Run evaluation on selected model and return results."""
    import subprocess
    import sys
    
    if not checkpoint_path:
        return "Please select a checkpoint first.", None, None, None, ""
    
    checkpoint_dir = Path(checkpoint_path).parent
    
    # Get category from checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        category = checkpoint.get('args', {}).get('category', 'bottle')
    except:
        category = 'bottle'
    
    # Determine correct data directory
    if (Path("./data") / category / "train").exists():
        data_dir = "./data"
    elif (Path("./data/original") / category / "train").exists():
        data_dir = "./data/original"
    else:
        return f"Error: Could not find dataset for '{category}'.", None, None, None, ""
    
    status = f"""RUNNING EVALUATION - Kuldeep Choksi's Anomaly Detection System
{'='*60}

Model: {checkpoint_path}
Category: {category}
Data Directory: {data_dir}

Running evaluation on test set...
"""
    
    # Build command
    cmd = [
        sys.executable, "evaluate.py",
        "--checkpoint", checkpoint_path,
        "--data-dir", data_dir,
        "--category", category
    ]
    
    try:
        # Run evaluation
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        output_lines = []
        for line in process.stdout:
            output_lines.append(line.strip())
        
        process.wait()
        
        eval_output = '\n'.join(output_lines)
        
        status += f"""
{'='*60}
EVALUATION OUTPUT:
{'='*60}

{eval_output}

{'='*60}
EVALUATION COMPLETE

Results are now available below.
"""
        
        # Load the generated results
        roc_img, dist_img, recon_img, summary = on_view_results(checkpoint_path)
        
        return status, roc_img, dist_img, recon_img, summary
        
    except Exception as e:
        return f"Error running evaluation: {str(e)}", None, None, None, ""


# ============================================================================
# BUILD THE UI
# ============================================================================

def create_ui():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="Anomaly Detection System") as app:
        
        # Header
        gr.Markdown("""
        # Industrial Anomaly Detection System
        **Developed by Kuldeep Choksi**
        
        This application detects manufacturing defects in images using deep learning.
        The model learns what "normal" looks like, then flags anything that deviates from normal patterns.
        
        ---
        
        **Getting Started:**
        1. If you have a trained model, go to **Analyze Images** tab
        2. If you need to train a model first, go to **Train Model** tab
        3. To view past results, go to **View Results** tab
        """)
        
        with gr.Tabs():
            
            # =================================================================
            # TAB 1: ANALYZE IMAGES
            # =================================================================
            with gr.Tab("Analyze Images"):
                gr.Markdown("""
                ## Analyze Images for Anomalies
                *Module developed by Kuldeep Choksi*
                
                Upload an image to check if it contains defects. The system will:
                - Reconstruct the image using the learned normal patterns
                - Generate an error heatmap showing where anomalies may exist
                - Provide an anomaly score and interpretation
                
                ---
                """)
                
                # Model Setup Section
                gr.Markdown("### Step 1: Load a Model")
                with gr.Row():
                    with gr.Column(scale=3):
                        checkpoint_dropdown = gr.Dropdown(
                            choices=get_available_checkpoints(),
                            label="Select Trained Model",
                            info="Choose a previously trained model checkpoint"
                        )
                    with gr.Column(scale=1):
                        refresh_btn = gr.Button("Refresh List", variant="secondary")
                        load_btn = gr.Button("Load Model", variant="primary")
                
                model_status = gr.Textbox(
                    label="Model Status",
                    lines=6,
                    value="No model loaded.\n\nSelect a checkpoint above and click 'Load Model' to begin.\n\nIf no checkpoints appear, go to the 'Train Model' tab first."
                )
                
                gr.Markdown("---")
                gr.Markdown("### Step 2: Upload and Analyze Image")
                
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(
                            label="Upload Image",
                            type="pil",
                            sources=["upload", "clipboard"]
                        )
                        analyze_btn = gr.Button("Analyze Image", variant="primary", size="lg")
                        
                        gr.Markdown("""
                        **Tips:**
                        - Use images similar to training data
                        - Supported formats: PNG, JPG, JPEG
                        - Image will be resized to 256x256
                        """)
                    
                    with gr.Column():
                        reconstruction_output = gr.Image(label="Reconstruction")
                    
                    with gr.Column():
                        heatmap_output = gr.Image(label="Error Heatmap")
                
                analysis_report = gr.Textbox(
                    label="Analysis Report",
                    lines=20,
                    value="Upload an image and click 'Analyze Image' to see results here."
                )
                
                # Wire up callbacks
                refresh_btn.click(
                    fn=lambda: gr.Dropdown(choices=get_available_checkpoints()),
                    outputs=checkpoint_dropdown
                )
                load_btn.click(fn=on_load_model, inputs=checkpoint_dropdown, outputs=model_status)
                analyze_btn.click(
                    fn=on_analyze_image,
                    inputs=input_image,
                    outputs=[reconstruction_output, heatmap_output, analysis_report]
                )
            
            # =================================================================
            # TAB 2: TRAIN MODEL
            # =================================================================
            with gr.Tab("Train Model"):
                gr.Markdown("""
                ## Train a New Model
                *Training pipeline developed by Kuldeep Choksi*
                
                Train an anomaly detection model on your dataset. The model will learn
                to reconstruct normal images, enabling it to detect anomalies as reconstruction failures.
                
                ---
                
                ### Prerequisites
                
                Before training, ensure you have:
                1. Downloaded the MVTec AD dataset (or your own data)
                2. Extracted it to `./data/original/` folder
                3. Your data folder structure should look like:
                ```
                data/original/
                    bottle/
                        train/good/
                        test/good/
                        test/broken_large/
                        ...
                ```
                
                ---
                """)
                
                gr.Markdown("### Training Configuration")
                
                with gr.Row():
                    with gr.Column():
                        category_dropdown = gr.Dropdown(
                            choices=get_available_categories(),
                            value="bottle",
                            label="Dataset Category",
                            info="Select which type of object to train on"
                        )
                        epochs_slider = gr.Slider(
                            minimum=10, maximum=200, value=50, step=10,
                            label="Number of Epochs",
                            info="More epochs = better learning, but longer training"
                        )
                        batch_slider = gr.Slider(
                            minimum=4, maximum=64, value=16, step=4,
                            label="Batch Size",
                            info="Lower if you run out of memory"
                        )
                    
                    with gr.Column():
                        lr_dropdown = gr.Dropdown(
                            choices=[0.0001, 0.0005, 0.001, 0.005],
                            value=0.001,
                            label="Learning Rate",
                            info="0.001 is a good default"
                        )
                        loss_dropdown = gr.Dropdown(
                            choices=["mse", "ssim", "combined"],
                            value="mse",
                            label="Loss Function",
                            info="MSE recommended - performs best on MVTec"
                        )
                
                gr.Markdown("---")
                
                train_btn = gr.Button("Start Training", variant="primary", size="lg")
                
                gr.Markdown("""
                **Note:** Training will run in the foreground. The interface may be unresponsive
                until training completes. Check the terminal for live progress.
                """)
                
                training_output = gr.Textbox(
                    label="Training Status",
                    lines=25,
                    value="Configure your training parameters above and click 'Start Training'.\n\nEstimated time: 25-50 minutes for 50 epochs on Apple Silicon."
                )
                
                # Wire up callbacks
                train_btn.click(
                    fn=on_start_training,
                    inputs=[category_dropdown, epochs_slider, batch_slider, lr_dropdown, loss_dropdown],
                    outputs=training_output
                )
            
            # =================================================================
            # TAB 3: VIEW RESULTS
            # =================================================================
            with gr.Tab("View Results"):
                gr.Markdown("""
                ## View Evaluation Results
                *Evaluation metrics by Kuldeep Choksi*
                
                Examine the performance metrics and visualizations from trained models.
                
                ---
                """)
                
                gr.Markdown("### Step 1: Select a Model")
                with gr.Row():
                    results_checkpoint = gr.Dropdown(
                        choices=get_available_checkpoints(),
                        label="Select Model to Evaluate"
                    )
                    refresh_results_btn = gr.Button("Refresh List", variant="secondary")
                
                gr.Markdown("### Step 2: Run Evaluation")
                gr.Markdown("""
                Click the button below to run evaluation on the test set. 
                This will compute AUROC and generate visualizations.
                """)
                
                with gr.Row():
                    run_eval_btn = gr.Button("Run Evaluation", variant="primary")
                    view_results_btn = gr.Button("View Existing Results", variant="secondary")
                
                eval_status = gr.Textbox(
                    label="Evaluation Status",
                    lines=15,
                    value="Select a model and click 'Run Evaluation' to generate metrics.\n\nOr click 'View Existing Results' if you've already run evaluation."
                )
                
                gr.Markdown("### Step 3: View Results")
                results_summary = gr.Textbox(label="Results Summary", lines=8)
                
                with gr.Row():
                    roc_image = gr.Image(label="ROC Curve")
                    dist_image = gr.Image(label="Score Distribution")
                
                recon_image = gr.Image(label="Sample Reconstructions")
                
                # Wire up callbacks
                refresh_results_btn.click(
                    fn=lambda: gr.Dropdown(choices=get_available_checkpoints()),
                    outputs=results_checkpoint
                )
                run_eval_btn.click(
                    fn=on_run_evaluation,
                    inputs=results_checkpoint,
                    outputs=[eval_status, roc_image, dist_image, recon_image, results_summary]
                )
                view_results_btn.click(
                    fn=on_view_results,
                    inputs=results_checkpoint,
                    outputs=[roc_image, dist_image, recon_image, results_summary]
                )
            
            # =================================================================
            # TAB 4: HELP
            # =================================================================
            with gr.Tab("Help"):
                gr.Markdown("""
                ## Help and Documentation
                *Documentation by Kuldeep Choksi*
                
                ---
                
                ### What is Anomaly Detection?
                
                Anomaly detection identifies data points that differ significantly from the majority.
                In manufacturing, this means finding defects in products without needing examples of
                every possible defect type.
                
                **How it works:**
                1. Train a model on ONLY normal (defect-free) images
                2. The model learns to reconstruct normal patterns
                3. When shown a defective image, reconstruction fails
                4. High reconstruction error = anomaly detected
                
                ---
                
                ### Using Your Own Dataset
                
                You can train on your own images! Follow this folder structure:
                
                ```
                data/
                    your_category_name/
                        train/
                            good/
                                image001.png
                                image002.png
                                ... (all your NORMAL images)
                        test/
                            good/
                                normal_test_001.png
                                ... (normal test images)
                            defect_type_1/
                                defect_001.png
                                ... (images with this defect)
                            defect_type_2/
                                another_defect_001.png
                                ...
                        ground_truth/  (optional)
                            defect_type_1/
                                defect_001_mask.png
                            defect_type_2/
                                another_defect_001_mask.png
                ```
                
                **Step-by-step for custom data:**
                
                1. Create folder: `data/my_product/`
                2. Create `data/my_product/train/good/` and put ALL your normal images there
                3. Create `data/my_product/test/good/` with some normal test images
                4. Create `data/my_product/test/defect_name/` for each defect type with defective images
                5. Restart the app - your category will appear in the dropdown
                6. Train and evaluate!
                
                **Example for optical fiber inspection:**
                ```
                data/
                    optical_fiber/
                        train/
                            good/
                                fiber_001.png
                                fiber_002.png
                                ... (500+ normal fiber images)
                        test/
                            good/
                                fiber_test_001.png (normal samples)
                            crack/
                                fiber_crack_001.png
                            contamination/
                                fiber_dirty_001.png
                ```
                
                **Tips for best results:**
                - Use at least 100+ training images (more is better)
                - Keep lighting and positioning consistent
                - Training images must be DEFECT-FREE
                - Test images should include both normal and defective samples
                - Image formats: PNG, JPG, JPEG
                - Images will be resized to 256x256 automatically
                
                ---
                
                ### Interpreting Results
                
                **AUROC (Area Under ROC Curve):**
                - 0.5 = Random guessing
                - 0.7-0.8 = Acceptable
                - 0.8-0.9 = Good
                - 0.9+ = Excellent
                
                **Anomaly Score:**
                - Lower = More normal
                - Higher = More anomalous
                - Compare to threshold to make decisions
                
                **Error Heatmap:**
                - Red/Yellow = High reconstruction error
                - Dark = Low reconstruction error
                - Anomalies show as bright spots
                
                ---
                
                ### Troubleshooting
                
                **My category doesn't appear in dropdown:**
                - Check folder structure matches the format above
                - Make sure `train/good/` folder exists with images
                - Restart the app after adding new data
                
                **No checkpoints available:**
                - Train a model first in the 'Train Model' tab
                
                **Training is slow:**
                - Reduce batch size
                - Reduce number of epochs
                - Ensure you're using GPU (check terminal output)
                
                **Poor detection results (low AUROC):**
                - Train for more epochs (try 100+)
                - Add more training images
                - Ensure training images are truly defect-free
                - Some defect types are inherently harder to detect
                
                **Out of memory:**
                - Reduce batch size to 4 or 8
                - Close other applications
                
                ---
                
                ### File Locations
                
                - **Your datasets:** `./data/<category_name>/`
                - **Trained models:** `./results/<category>_<timestamp>/`
                - **Evaluation outputs:** `./results/<category>_<timestamp>/evaluation/`
                
                ---
                
                ### Command Line Usage
                
                You can also run training and evaluation from the terminal:
                
                ```bash
                # Train on custom dataset
                python train.py --category optical_fiber --data-dir ./data --epochs 100
                
                # Train on MVTec dataset
                python train.py --category bottle --data-dir ./data/original --epochs 50
                
                # Evaluate  
                python evaluate.py --checkpoint results/optical_fiber_xxx/best_model.pth --data-dir ./data
                ```
                
                ---
                
                ### About
                
                This anomaly detection system was developed by **Kuldeep Choksi** as part of 
                a computer vision portfolio project. The system uses convolutional autoencoders 
                to learn normal patterns and detect manufacturing defects.
                
                **Technical Stack:**
                - PyTorch for deep learning
                - Gradio for the user interface
                - MVTec AD dataset for benchmarking
                
                **Contact:** [GitHub](https://github.com/KuldeepChoksi)
                """)
        
        # Footer
        gr.Markdown("""
        ---
        *Anomaly Detection System | Developed by Kuldeep Choksi | Built with PyTorch and Gradio*
        """)
    
    return app


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("ANOMALY DETECTION SYSTEM")
    print("Developed by Kuldeep Choksi")
    print("="*60)
    print()
    print("Starting application...")
    print()
    print("Once loaded, open your browser to: http://localhost:7860")
    print()
    print("Press Ctrl+C to stop the server.")
    print("="*60)
    
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )