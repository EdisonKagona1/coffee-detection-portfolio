"""
Quantitative Validation and Model Comparison Script.

This script evaluates one or more trained YOLOv8 models on a practical
counting task. It compares the number of detected objects against a
ground-truth count provided in a CSV file and reports key error metrics.

This script is the professional replacement for the quantitative analysis cells
in the `post_training_analysis.ipynb` notebook.

Example Usage:
    # Validate a single model
    python validate.py --model_paths runs/train/exp1/weights/best.pt --csv_path data/validation_counts.csv --images_dir data/validation_images

    # Compare multiple models
    python validate.py --model_paths runs/train/exp1/weights/best.pt runs/train/exp2/weights/best.pt --csv_path ... --images_dir ...
"""
import cv2
import argparse
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

def validate_model_on_counting_task(model: YOLO, df: pd.DataFrame, images_dir: Path) -> dict:
    """
    Tests a model against a DataFrame with ground-truth counts.

    Args:
        model (YOLO): The loaded YOLOv8 model object.
        df (pd.DataFrame): DataFrame containing ground-truth data. Must have
                           'image_id' and 'ground_truth_count' columns.
        images_dir (Path): The directory where the test images are stored.

    Returns:
        dict: A dictionary containing the calculated error metrics.
    """
    total_absolute_error = 0
    total_percentage_error = 0
    images_tested = 0
    images_missing = 0

    # Use tqdm for a nice progress bar during validation
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Validating images"):
        image_id = row['image_id']
        ground_truth_count = row['ground_truth_count']

        image_path = images_dir / f"{image_id}.jpg"

        if not image_path.exists():
            images_missing += 1
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            images_missing += 1
            continue
        
        # Run prediction on the image
        # Set verbose=False to keep the output clean
        prediction = model.predict(img, save=False, verbose=False, conf=0.4)
        
        # The number of detected boxes is the predicted count
        predicted_count = len(prediction[0].boxes)

        # Calculate errors for this image
        absolute_error = abs(predicted_count - ground_truth_count)
        percentage_error = absolute_error / ground_truth_count if ground_truth_count > 0 else 0

        total_absolute_error += absolute_error
        total_percentage_error += percentage_error
        images_tested += 1
    
    # Calculate average metrics
    mean_absolute_error = total_absolute_error / images_tested if images_tested > 0 else 0
    mean_percentage_error = total_percentage_error / images_tested if images_tested > 0 else 0
    
    print(f"Validation complete. Tested {images_tested} images. Skipped {images_missing} missing images.")
    
    return {
        "mean_absolute_error": mean_absolute_error,
        "mean_percentage_error": mean_percentage_error,
        "images_tested": images_tested
    }


def main(args: argparse.Namespace) -> None:
    """
    Main function to load models and the validation data, then print a comparison table.
    """
    try:
        # Assuming the CSV has columns named 'image_id' and 'ground_truth_count'
        validation_df = pd.read_csv(args.csv_path, sep=args.csv_separator)
    except FileNotFoundError:
        print(f"Error: Ground-truth CSV file not found at '{args.csv_path}'")
        return
    except KeyError as e:
        print(f"Error: CSV file must contain the columns 'image_id' and 'ground_truth_count'. Missing {e}.")
        return

    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        print(f"Error: Images directory not found at '{args.images_dir}'")
        return

    # --- Optional: Download logic would go here ---
    # You could add logic to use runcmd from utils.py to wget images if they don't exist
    
    results = []
    for model_path_str in args.model_paths:
        model_path = Path(model_path_str)
        if not model_path.exists():
            print(f"Warning: Model not found at '{model_path_str}', skipping.")
            continue
        
        print(f"\n--- Evaluating model: {model_path.parent.name}/{model_path.name} ---")
        model = YOLO(model_path)
        
        metrics = validate_model_on_counting_task(model, validation_df, images_dir)
        
        # Use the name of the model's parent directory for a clean label
        model_name = model_path.parent.parent.name
        metrics['model_name'] = model_name
        results.append(metrics)

    if not results:
        print("No models were evaluated.")
        return

    # Create and print a clean comparison table using Pandas
    results_df = pd.DataFrame(results)
    results_df = results_df[['model_name', 'mean_absolute_error', 'mean_percentage_error', 'images_tested']]
    results_df = results_df.rename(columns={
        "model_name": "Model",
        "mean_absolute_error": "Mean Absolute Error",
        "mean_percentage_error": "Mean Percentage Error (%)"
    })
    results_df['Mean Percentage Error (%)'] = (results_df['Mean Percentage Error (%)'] * 100).map('{:.2f}%'.format)
    results_df['Mean Absolute Error'] = results_df['Mean Absolute Error'].map('{:.2f}'.format)

    print("\n--- Final Model Comparison ---")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and compare YOLO models on a quantitative counting task.")
    
    parser.add_argument(
        '--model-paths', 
        nargs='+',  # This allows for one or more model paths
        required=True, 
        help='List of paths to trained model weights (.pt files), separated by spaces.'
    )
    parser.add_argument(
        '--csv-path', 
        type=str, 
        required=True, 
        help="Path to the ground-truth CSV file. Must contain 'image_id' and 'ground_truth_count' columns."
    )
    parser.add_argument(
        '--images-dir', 
        type=str, 
        required=True, 
        help='Path to the directory containing the validation images.'
    )
    parser.add_argument(
        '--csv-separator', 
        type=str, 
        default=',', 
        help='The separator used in the CSV file (e.g., "," or ";"). Default is a comma.'
    )
    
    args = parser.parse_args()
    main(args)