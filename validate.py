"""
Quantitative Validation, Correction, and Model Comparison Script.

This script evaluates trained YOLOv8 models on a practical counting task and
demonstrates a two-stage modeling approach by training a polynomial regression
model to correct for systematic underestimation caused by occlusion.

It reports both the raw YOLO model error and the corrected error.

This script is the professional replacement for the advanced analysis in the
`post_training_analysis.ipynb` notebook.

Example Usage:
    # Validate and apply correction for one model
    python validate.py --model-paths runs/train/exp1/weights/best.pt --csv_path data/validation_counts.csv --images_dir data/validation_images
"""
import cv2
import argparse
import pandas as pd
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def get_yolo_predictions(model: YOLO, df: pd.DataFrame, images_dir: Path) -> pd.DataFrame:
    """
    Runs YOLO inference on a set of images and returns a DataFrame with predictions.
    """
    predictions = []
    print("Step 1: Generating raw predictions from YOLO model...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="YOLO Inference"):
        image_id = row['image_id']
        image_path = images_dir / f"{image_id}.jpg"

        if not image_path.exists():
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            continue
        
        # Run YOLO prediction
        prediction = model.predict(img, save=False, verbose=False, conf=0.4)
        predicted_count = len(prediction[0].boxes)
        
        predictions.append({'image_id': image_id, 'predicted_count': predicted_count})

    return pd.merge(df, pd.DataFrame(predictions), on='image_id')


def train_and_apply_correction_model(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Trains a polynomial regression model to correct YOLO's predictions and applies it.
    """
    print("\nStep 2: Training a polynomial error correction model...")
    
    # Use 80% of the data to train the correction model, 20% to test it
    train_df, test_df = train_test_split(results_df, test_size=0.2, random_state=42)

    X_train = train_df[['predicted_count']]
    y_train = train_df['ground_truth_count']

    # Create polynomial features (e.g., predicted_count^2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)

    # Train the linear regression model on these polynomial features
    correction_model = LinearRegression()
    correction_model.fit(X_train_poly, y_train)
    print("Correction model trained.")

    # Apply the correction to the full dataset
    X_full_poly = poly.transform(results_df[['predicted_count']])
    corrected_predictions = correction_model.predict(X_full_poly)
    
    # Add corrected predictions to the DataFrame, ensuring they are integers
    results_df['corrected_count'] = np.round(corrected_predictions).astype(int)
    
    return results_df


def main(args: argparse.Namespace) -> None:
    """
    Main function to run the full validation and correction pipeline.
    """
    try:
        validation_df = pd.read_csv(args.csv_path, sep=args.csv_separator)
        # Ensure required columns exist
        _ = validation_df['image_id']
        _ = validation_df['ground_truth_count']
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading or parsing CSV file: {e}")
        return

    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        print(f"Error: Images directory not found at '{args.images_dir}'")
        return

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model not found at '{args.model_path}'")
        return
        
    print(f"\n--- Evaluating model: {model_path.name} ---")
    model = YOLO(model_path)
    
    # Step 1: Get raw predictions
    results_with_preds = get_yolo_predictions(model, validation_df, images_dir)
    
    if results_with_preds.empty:
        print("Could not generate any predictions. Please check image paths and data.")
        return

    # Step 2: Train and apply the correction model
    final_results = train_and_apply_correction_model(results_with_preds)
    
    # Step 3: Calculate and report final metrics
    print("\nStep 3: Calculating final performance metrics...")
    
    raw_mae = mean_absolute_error(final_results['ground_truth_count'], final_results['predicted_count'])
    corrected_mae = mean_absolute_error(final_results['ground_truth_count'], final_results['corrected_count'])
    
    print("\n--- Final Validation Report ---")
    print(f"Model: {model_path.name}")
    print("-" * 30)
    print(f"Raw YOLO Model MAE:         {raw_mae:.4f}")
    print(f"Corrected Two-Stage MAE:    {corrected_mae:.4f}")
    print("-" * 30)
    improvement = raw_mae - corrected_mae
    improvement_percent = (improvement / raw_mae) * 100
    print(f"Improvement from Correction Model: {improvement:.4f} ({improvement_percent:.2f}%)")
    print("\n✅ Validation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a YOLO model and apply a polynomial error correction.")
    
    parser.add_argument('--model-path', type=str, required=True, help='Path to a single trained model weights (.pt file).')
    parser.add_argument('--csv-path', type=str, required=True, help="Path to the ground-truth CSV file. Must contain 'image_id' and 'ground_truth_count' columns.")
    parser.add_argument('--images-dir', type=str, required=True, help='Path to the directory containing the validation images.')
    parser.add_argument('--csv-separator', type=str, default=',', help='The separator used in the CSV file (e.g., "," or ";").')
    
    args = parser.parse_args()
    main(args)