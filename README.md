
# Video Anomaly Detection

A deep learning pipeline for detecting anomalies in video sequences using autoencoder-based reconstruction and temporal modeling.

## Overview

This project implements an unsupervised anomaly detection system that learns to model "normal" patterns in video data and identifies deviations as anomalies. The approach is applicable to industrial inspection, surveillance, and medical imaging domains.

## Approach

1. **Spatial Feature Learning**: Convolutional autoencoder learns to reconstruct normal frames
2. **Temporal Modeling**: ConvLSTM captures temporal dependencies across frame sequences
3. **Anomaly Scoring**: Reconstruction error serves as anomaly score - high error indicates anomaly
4. **Localization**: Pixel-wise error heatmaps show where anomalies occur

## Project Structure

```
video-anomaly-detection/
├── data/               # Dataset storage
├── models/             # Neural network architectures
│   ├── autoencoder.py  # Convolutional autoencoder
│   └── convlstm.py     # Temporal modeling
├── utils/              # Helper functions
│   ├── dataset.py      # Data loading utilities
│   ├── preprocessing.py# Frame preprocessing
│   └── metrics.py      # Evaluation metrics
├── notebooks/          # Exploration and visualization
├── results/            # Outputs and visualizations
├── train.py            # Training script
├── evaluate.py         # Evaluation script
└── main.py             # Entry point
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

Using MVTec Anomaly Detection dataset - industry standard benchmark for anomaly detection.

## Usage

```bash
# Train the model
python train.py --dataset mvtec --category bottle

# Evaluate
python evaluate.py --checkpoint results/best_model.pth
```

## Results

Coming soon...

## References

- MVTec AD Dataset: https://www.mvtec.com/company/research/datasets/mvtec-ad
- Autoencoder-based anomaly detection literature
