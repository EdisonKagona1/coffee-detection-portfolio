"""
Active Inference Validator for Shared Autonomy Simulation.

This script simulates a human-in-the-loop workflow. It uses uncertainty
estimates from an MC Dropout model (`mc_inference.py`) to decide when
to "ask a human" for a correct label, demonstrating a core concept of
active inference and shared autonomy.

The goal is to achieve the highest accuracy while minimizing the number of
human queries, thus optimizing an expert's time.

Example Usage:
    python active_validator.py --weights yolov8s.pt --images-dir path/to/images --csv-path path/to/counts.csv
"""
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

# We will re-use the core logic from mc_inference.py by importing it.
# This is excellent code practice.
try:
    from mc_inference import mc_dropout_inference
except ImportError:
    print("Error: Could not import 'mc_dropout_inference' from mc_inference.py.")
    print("Please ensure mc_inference.py is in the same directory.")
    exit()

def simulate_shared_autonomy(
    df: pd.DataFrame, 
    images_dir: Path, 
    weights_path: str, 
    uncertainty_threshold: float,
    mc_samples: int
) -> dict:
    """
    Simulates the active validation loop.

    Args:
        df (pd.DataFrame): DataFrame with 'image_id' and 'ground_truth_count'.
        images_dir (Path): Directory containing the images.
        weights_path (str): Path to the trained model.
        uncertainty_threshold (float): The uncertainty score above which the system will "query the human".
        mc_samples (int): Number of MC samples for uncertainty estimation.

    Returns:
        A dictionary summarizing the simulation results.
    """
    total_images_processed = 0
    total_human_queries = 0
    total_absolute_error = 0
    
    print("\n--- Starting Active Validation Simulation ---")
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Simulating"):
        image_id = row['image_id']
        ground_truth_count = row['ground_truth_count']
        image_path = images_dir / f"{image_id}.jpg"

        if not image_path.exists():
            continue

        # Step 1: Get predictions AND uncertainty from our MC Dropout script's function
        inference_results = mc_dropout_inference(weights_path, str(image_path), mc_samples)

        if not inference_results or len(inference_results['mean_boxes']) == 0:
            model_prediction = 0
            is_uncertain = False
        else:
            average_uncertainty = np.mean(inference_results['uncertainties'])
            model_prediction = len(inference_results['mean_boxes'])
            
            # Step 2: The Active Inference Decision
            if average_uncertainty > uncertainty_threshold:
                is_uncertain = True
            else:
                is_uncertain = False

        # Step 3: Simulate the outcome
        if is_uncertain:
            # The system "queries the human" and gets the perfect ground truth answer.
            final_count = ground_truth_count
            total_human_queries += 1
        else:
            # The system is confident and "commits" to its own prediction.
            final_count = model_prediction
        
        total_absolute_error += abs(final_count - ground_truth_count)
        total_images_processed += 1
    
    if total_images_processed == 0:
        return None

    # Calculate final metrics
    final_mae = total_absolute_error / total_images_processed
    query_rate = (total_human_queries / total_images_processed) * 100

    return {
        "final_mae": final_mae,
        "query_rate_percent": query_rate,
        "total_images": total_images_processed
    }

def main(args: argparse.Namespace) -> None:
    """
    Main function to run the simulation and print a report.
    """
    try:
        validation_df = pd.read_csv(args.csv_path, sep=args.csv_separator)
        _ = validation_df['image_id']
        _ = validation_df['ground_truth_count']
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading or parsing CSV file: {e}")
        return

    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        print(f"Error: Images directory not found at '{args.images_dir}'")
        return

    results = simulate_shared_autonomy(
        validation_df, 
        images_dir, 
        args.weights, 
        args.threshold, 
        args.samples
    )
    
    if results:
        print("\n--- Active Inference Simulation Report ---")
        print(f"Model: {args.weights}")
        print(f"Uncertainty Threshold: {args.threshold}")
        print("-" * 40)
        print(f"Final System Mean Absolute Error: {results['final_mae']:.4f}")
        print(f"Human Query Rate: {results['query_rate_percent']:.2f}%")
        print(f"Total Images Processed: {results['total_images']}")
        print("-" * 40)
        print("This result shows the final accuracy of a system that intelligently asks for help when it is uncertain.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a shared autonomy system using active inference.")
    
    parser.add_argument('--weights', type=str, required=True, help="Path to trained model weights (.pt) or a standard name like 'yolov8s.pt'.")
    parser.add_argument('--images-dir', type=str, required=True, help='Path to the directory containing the validation images.')
    parser.add_argument('--csv-path', type=str, required=True, help="Path to the ground-truth CSV file. Must contain 'image_id' and 'ground_truth_count' columns.")
    parser.add_argument('--threshold', type=float, default=50.0, help="Uncertainty score threshold to trigger a 'human query'.")
    # THE FIX IS HERE:
    parser.add_argument('--samples', type=int, default=15, help='Number of MC samples to run for uncertainty estimation.')
    parser.add_argument('--csv-separator', type=str, default=',', help='The separator used in the CSV file (e.g., "," or ";").')
    
    args = parser.parse_args()
    main(args)