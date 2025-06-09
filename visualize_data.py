"""
Data Visualization and Label Verification Script.

This script provides utilities to visualize images from the coffee bean dataset
and their corresponding YOLO-format bounding box labels. It helps ensure data
integrity before training a model.

This script is the professional replacement for the exploratory `check_pre_process.ipynb` notebook.

Example Usage:
    # Display 5 random images from a data directory
    python visualize_data.py --data_dir path/to/your/data --num_images 5

    # Save visualizations for all images in a directory without showing them
    python visualize_data.py --data_dir path/to/your/data --num_images -1 --save --no-display
"""
import cv2
import random
import argparse
from pathlib import Path

# Import the custom visualization function from our toolkit
try:
    from utils import display_image_and_box
except ImportError:
    print("Error: Could not import 'display_image_and_box' from utils.py.")
    print("Please ensure utils.py is in the same directory.")
    exit()


def visualize_dataset(data_dir: str, num_images: int, save: bool, display: bool) -> None:
    """
    Loads images and their corresponding YOLO labels from a directory and
    displays or saves the visualizations.

    Args:
        data_dir (str): The path to the directory containing images and labels.
        num_images (int): The number of random images to process.
                          If -1, all images in the directory are processed.
        save (bool): If True, saves visualizations to the 'outputs/label_verification' directory.
        display (bool): If True, shows the visualizations on screen via Matplotlib.
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        print(f"Error: Directory not found at '{data_dir}'")
        return

    # Use pathlib's glob to robustly find all common image types
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_files = [p for ext in image_extensions for p in data_path.glob(ext)]

    if not image_files:
        print(f"Error: No images (.jpg, .jpeg, .png) found in '{data_dir}'")
        return
    
    print(f"Found {len(image_files)} images in the directory.")

    # Determine which images to process based on user input
    if num_images != -1 and num_images < len(image_files):
        random.shuffle(image_files)
        images_to_process = image_files[:num_images]
    else:
        images_to_process = sorted(image_files)

    print(f"Processing {len(images_to_process)} images...")

    for image_path in images_to_process:
        label_path = image_path.with_suffix('.txt')

        # This check prevents the FileNotFoundError from your notebook
        if not label_path.exists():
            print(f"Warning: Label file not found for '{image_path.name}', skipping.")
            continue

        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise IOError("Image file could not be read or is corrupted.")

            with open(label_path, 'r') as bbox_file:
                display_image_and_box(
                    img,
                    bbox_file,
                    name=image_path.name,
                    save=save,
                    display=display
                )

        except Exception as e:
            print(f"An error occurred while processing '{image_path.name}': {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize YOLO object detection data by drawing labels on images.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the directory containing images and their .txt label files."
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=5,
        help="Number of random images to process. \nUse -1 to process all images. (default: 5)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="If set, saves output images to 'outputs/label_verification'."
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="If set, prevents images from being displayed on screen."
    )

    args = parser.parse_args()
    display_on_screen = not args.no_display
    visualize_dataset(args.data_dir, args.num_images, args.save, display_on_screen)