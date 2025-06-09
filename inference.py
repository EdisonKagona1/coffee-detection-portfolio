"""
Inference Script for YOLOv8.

This script runs a trained YOLOv8 model on a directory of images or a single
image file and saves the annotated results. It is a simple tool for visually
inspecting model predictions on new data.

This replaces the basic prediction cells from `post_training_analysis.ipynb`.

Example Usage:
    # Run inference on a directory of images
    python inference.py --weights runs/train/exp1/weights/best.pt --source data/unseen_images/

    # Run inference on a single image
    python inference.py --weights runs/train/exp1/weights/best.pt --source assets/sample.jpg
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

def run_inference(weights_path: str, source_path: str, project_name: str, run_name: str) -> None:
    """
    Loads a trained model and runs inference on a given source.

    Args:
        weights_path (str): Path to the trained model weights (.pt file).
        source_path (str): Path to the source image or directory of images.
        project_name (str): The root directory to save results.
        run_name (str): The specific name for this inference run.
    """
    model_path = Path(weights_path)
    source = Path(source_path)

    if not model_path.exists():
        print(f"Error: Model weights not found at '{weights_path}'")
        return

    if not source.exists():
        print(f"Error: Inference source not found at '{source_path}'")
        return

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"--- Running Inference ---")
    print(f"  Model: {weights_path}")
    print(f"  Source: {source_path}")
    print("-------------------------")

    try:
        # The `predict` method can handle both single images and directories
        model.predict(
            source=source,
            save=True,          # Save images with bounding boxes
            project=project_name,
            name=run_name,
            exist_ok=True,      # Overwrite results from a previous run with the same name
            line_thickness=2,   # Make bounding boxes a bit thicker
            conf=0.4,           # Only show detections with confidence > 0.4
        )
    except Exception as e:
        print(f"An error occurred during inference: {e}")
        return

    print(f"\n✅ Inference complete.")
    print(f"Results saved to '{project_name}/{run_name}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference on images or directories.")
    
    parser.add_argument(
        '--weights', 
        type=str, 
        required=True, 
        help='Path to the trained model weights (.pt file).'
    )
    parser.add_argument(
        '--source', 
        type=str, 
        required=True, 
        help='Path to the source image or directory of images.'
    )
    parser.add_argument(
        '--project', 
        type=str, 
        default='runs/detect', 
        help='Project directory where results will be saved.'
    )
    parser.add_argument(
        '--name', 
        type=str, 
        default='inference_run', 
        help='Name for the inference run directory.'
    )
    
    args = parser.parse_args()
    run_inference(args.weights, args.source, args.project, args.name)