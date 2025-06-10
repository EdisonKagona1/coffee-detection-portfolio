"""
Monte Carlo Dropout Inference Script for Uncertainty Estimation.

This script loads a trained YOLOv8 model and performs N stochastic forward passes
with dropout layers activated to approximate the model's uncertainty for each
detection.

The output is a set of statistics (mean and variance) for each bounding box,
which can be used by an active learning or shared autonomy system.

Example Usage:
    # Use a standard, downloadable model
    python mc_inference.py --weights yolov8s.pt --source path/to/image.jpg --samples 30

    # Use a locally trained model
    python mc_inference.py --weights runs/train/exp1/weights/best.pt --source path/to/image.jpg
"""
import torch
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
from collections import Counter

def enable_dropout(model: torch.nn.Module):
    """
    Recursively sets dropout layers in a model to training mode.
    This is essential for Monte Carlo Dropout.
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def mc_dropout_inference(weights_path: str, source_path: str, num_samples: int) -> dict:
    """
    Performs Monte Carlo Dropout inference on a single image to estimate uncertainty.

    Args:
        weights_path (str): Path to trained weights or a standard model name (e.g., 'yolov8s.pt').
        source_path (str): Path to the source image.
        num_samples (int): The number of stochastic forward passes to perform.

    Returns:
        A dictionary containing the mean predictions and their calculated uncertainties.
        Returns None if a critical error occurs.
    """
    # --- Step 1: Validate Inputs and Load Model/Image ---
    source_path_obj = Path(source_path)
    if not source_path_obj.is_file():
        print(f"Error: Source image not found at: {source_path}")
        return None

    try:
        # Let the YOLO constructor handle model loading. It will download standard
        # models like 'yolov8s.pt' if they don't exist locally.
        model = YOLO(weights_path)
        img = cv2.imread(str(source_path_obj))
        if img is None:
            raise IOError("Could not read image file.")
    except Exception as e:
        print(f"Error loading model or image: {e}")
        return None

    print(f"Performing {num_samples} stochastic forward passes...")

    # --- Step 2: Core MC Dropout Logic ---
    enable_dropout(model.model)  # Set dropout layers to train() mode
    
    all_predictions = []
    # Use torch.no_grad() for efficiency as we don't need gradients
    with torch.no_grad():
        for _ in tqdm(range(num_samples), desc="MC Samples"):
            results = model(img, verbose=False, conf=0.2)  # Use a low confidence threshold
            # Store bounding boxes [x1, y1, x2, y2, conf, cls]
            all_predictions.append(results[0].boxes.data)

    if not all_predictions:
        print("No predictions were made.")
        return {'mean_boxes': [], 'uncertainties': []}

    # --- Step 3: Process Predictions to Calculate Uncertainty ---
    # This is a simplified approach to matching boxes across runs.
    # A production system might use a more advanced matching algorithm (e.g., Hungarian algorithm).
    box_counts = [len(p) for p in all_predictions]
    if not box_counts or max(box_counts) == 0:
        print("No objects detected in any of the samples.")
        return {'mean_boxes': [], 'uncertainties': []}
        
    most_common_n_boxes = Counter(box_counts).most_common(1)[0][0]
    
    # Filter for predictions that have the most common number of boxes for stability
    stable_predictions = [p for p in all_predictions if len(p) == most_common_n_boxes]
    
    if not stable_predictions:
        print("Could not form a stable set of predictions for uncertainty calculation.")
        return {'mean_boxes': [], 'uncertainties': []}

    # Stack the stable predictions into a single tensor for vectorized operations
    stacked_preds = torch.stack(stable_predictions)
    
    # Calculate statistics across the samples dimension (dim=0)
    mean_boxes = stacked_preds.mean(dim=0)
    variance_boxes = stacked_preds.var(dim=0)
    
    # Define uncertainty as the sum of variances of the 4 bounding box coordinates
    uncertainty_scores = variance_boxes[:, :4].sum(dim=1)

    print("✅ Uncertainty estimation complete.")
    return {
        'mean_boxes': mean_boxes.cpu().numpy(),
        'uncertainties': uncertainty_scores.cpu().numpy()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate model uncertainty for object detection using MC Dropout.")
    parser.add_argument('--weights', type=str, required=True, help="Path to trained model weights (.pt) or a standard model name like 'yolov8s.pt'.")
    parser.add_argument('--source', type=str, required=True, help='Path to the source image.')
    parser.add_argument('--samples', type=int, default=30, help='Number of Monte Carlo samples to run.')
    
    args = parser.parse_args()
    
    results = mc_dropout_inference(args.weights, args.source, args.samples)
    
    if results and len(results['mean_boxes']) > 0:
        print("\n--- Inference Results with Uncertainty ---")
        for i, (box, uncertainty) in enumerate(zip(results['mean_boxes'], results['uncertainties'])):
            x1, y1, x2, y2, conf, cls = box
            print(
                f"Detection {i+1}: "
                f"Class={int(cls)}, "
                f"Confidence={conf:.2f}, "
                f"Box=[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}], "
                f"Uncertainty Score={uncertainty:.4f}"
            )