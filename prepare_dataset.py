"""
Dataset Preparation and Preprocessing Script.

This script takes the raw coffee bean dataset (images and XML labels) and
processes it into the YOLOv5 format required for training. It handles:
1. Converting XML labels to YOLO .txt format.
2. Remapping class labels for different experimental setups.
3. Preprocessing images via a robust slicing-and-resizing strategy.
4. Splitting the final dataset into training and validation sets.

This script is the professional replacement for the data preparation cells
in the `Coffee_first_train.ipynb` notebook.

Example Usage:
    # Prepare the multi-class dataset
    python prepare_dataset.py --config multi-class

    # Prepare the single-class dataset for defect detection
    python prepare_dataset.py --config single-class
"""
import os
import cv2
import json
import shutil
import fnmatch
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict

import albumentations as A
import yaml
from tqdm import tqdm

# Import the utility functions we created
from utils import xml_to_yolo, calculate_slice_bboxes

# --- Core Data Processing Functions ---
# These are the specific functions for this pipeline.

def adapt_classes(labels_path: Path, classes_json_path: Path, config: str) -> None:
    """
    Remaps class labels in the .txt files based on the desired configuration.
    """
    with open(classes_json_path, 'r') as f:
        class_data = json.load(f)
    
    # Create a mapping from old index to new name
    old_idx_to_name = {str(cat['id']): cat['name'] for cat in class_data['categories']}
    
    if config == 'multi-class':
        # Filter out ignored classes and create a new mapping
        to_ignore = {"low_visibility_unsure"}
        final_classes = [name for name in old_idx_to_name.values() if name not in to_ignore]
        name_to_new_idx = {name: str(i) for i, name in enumerate(final_classes)}
    else: # single-class
        name_to_new_idx = None

    print(f"Adapting class labels for '{config}' configuration...")
    for bbox_txt_path in tqdm(list(labels_path.glob("*.txt"))):
        new_bbox_str = ""
        with open(bbox_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(" ")
                old_label_idx = parts[0]

                if config == 'single-class':
                    parts[0] = '0' # All defects become class 0
                    new_bbox_str += " ".join(parts) + "\n"
                elif config == 'multi-class':
                    class_name = old_idx_to_name.get(old_label_idx)
                    if class_name and class_name not in to_ignore:
                        parts[0] = name_to_new_idx[class_name]
                        new_bbox_str += " ".join(parts) + "\n"
        
        with open(bbox_txt_path, 'w') as f:
            f.write(new_bbox_str)

def process_raw_data(raw_data_dir: Path, processed_dir: Path, downsize: tuple = (480, 640)) -> None:
    """
    Main preprocessing function. Converts XML, handles problematic images,
    and applies the advanced slicing strategy.
    """
    raw_images_dir = raw_data_dir / "images"
    raw_xml_dir = raw_data_dir / "labels_xml" # Assuming XMLs are here
    processed_dir.mkdir(parents=True, exist_ok=True)

    problematic_images = {"1615315525581"} # Known bad data
    
    print("Processing raw images and labels...")
    for xml_path in tqdm(list(raw_xml_dir.glob("*.xml"))):
        img_name = xml_path.stem
        if img_name in problematic_images:
            continue

        image_path = raw_images_dir / f"{img_name}.jpg"
        if not image_path.exists():
            continue
            
        current_img = cv2.imread(str(image_path))
        if current_img is None:
            continue
            
        # Handle known orientation issues
        if img_name == "1615385610373": 
          current_img = cv2.rotate(current_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if img_name == "1615581114578": 
          current_img = cv2.rotate(current_img, cv2.ROTATE_180)

        img_h, img_w, _ = current_img.shape
        
        # Convert XML to YOLO format first
        xml_tree = ET.parse(xml_path)
        yolo_str = xml_to_yolo(xml_tree.findall("object"), img_w, img_h)

        # Parse the YOLO string into a list of boxes and labels
        bboxes = []
        labels = []
        for line in yolo_str.strip().split('\n'):
            if not line: continue
            parts = list(map(float, line.split(" ")))
            labels.append([int(parts[0])])
            bboxes.append(parts[1:])

        # Apply the advanced slicing strategy
        slice_coords = calculate_slice_bboxes(img_h, img_w, slice_height=1024, slice_width=768, overlap_height_ratio=0.05, overlap_width_ratio=0.05)

        for i, (xmin, ymin, xmax, ymax) in enumerate(slice_coords):
            crop_transform = A.Compose(
                [A.Crop(x_min=xmin, y_min=ymin, x_max=xmax, y_max=ymax)],
                bbox_params=A.BboxParams(format='yolo', label_fields=['labels'], min_visibility=0.1, min_area=10)
            )
            try:
                transformed = crop_transform(image=current_img, bboxes=bboxes, labels=labels)
            except ValueError:
                continue # Skip crops with invalid bboxes

            if len(transformed['bboxes']) > 0:
                new_img_name = f"{img_name}_c{i}"
                final_img = cv2.resize(transformed['image'], dsize=downsize)
                cv2.imwrite(str(processed_dir / f"{new_img_name}.jpg"), final_img)

                bbox_str = ""
                for label, bbox in zip(transformed['labels'], transformed['bboxes']):
                    bbox_str += f"{label[0]} {' '.join(map(str, bbox))}\n"
                
                with open(processed_dir / f"{new_img_name}.txt", "w") as f:
                    f.write(bbox_str)

def create_train_val_split(source_dir: Path, proportion_train: float = 0.85) -> None:
    """
    Splits the processed data into train and validation sets.
    """
    train_dir = source_dir / "train"
    val_dir = source_dir / "val"

    # Create directories
    (train_dir / "images").mkdir(parents=True, exist_ok=True)
    (train_dir / "labels").mkdir(parents=True, exist_ok=True)
    (val_dir / "images").mkdir(parents=True, exist_ok=True)
    (val_dir / "labels").mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(source_dir.glob("*.jpg")))
    shutil.os.random.shuffle(image_files)

    split_idx = int(len(image_files) * proportion_train)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    print(f"Splitting data: {len(train_files)} training, {len(val_files)} validation.")

    # Move files
    for file_set, target_dir in [(train_files, train_dir), (val_files, val_dir)]:
        for img_path in file_set:
            label_path = img_path.with_suffix('.txt')
            shutil.move(str(img_path), str(target_dir / "images" / img_path.name))
            if label_path.exists():
                shutil.move(str(label_path), str(target_dir / "labels" / label_path.name))

# --- Main Execution Logic ---

def main(args: argparse.Namespace) -> None:
    """
    Main function to run the data preparation pipeline based on the chosen config.
    """
    raw_data_dir = Path("data/raw")
    processed_data_dir = Path("data/processed")
    config_dir = Path("data/configs")

    # Clean up previous runs
    if processed_data_dir.exists():
        print(f"Removing old processed data directory: {processed_data_dir}")
        shutil.rmtree(processed_data_dir)
    
    # 1. Process Raw Data (XML to YOLO, slicing, etc.)
    intermediate_dir = processed_data_dir / "intermediate"
    process_raw_data(raw_data_dir, intermediate_dir)

    # 2. Adapt class labels based on config
    adapt_classes(intermediate_dir, raw_data_dir / "notes.json", args.config)

    # 3. Create train/val split
    final_dataset_dir = processed_data_dir / f"coffee_{args.config}"
    shutil.copytree(intermediate_dir, final_dataset_dir)
    create_train_val_split(final_dataset_dir)

    # 4. Create the final .yaml file for training
    config_dir.mkdir(exist_ok=True)
    yaml_path = config_dir / f"coffee_{args.config}.yaml"
    
    if args.config == 'multi-class':
        data_yaml = {
            'path': f"../data/processed/coffee_{args.config}", # Path relative to the YAML file
            'train': 'train/images',
            'val': 'val/images',
            'nc': 4,
            'names': ["dark_brown", "green_cherry", "red_cherry", "yellow_cherry"]
        }
    else: # single-class
        data_yaml = {
            'path': f"../data/processed/coffee_{args.config}",
            'train': 'train/images',
            'val': 'val/images',
            'nc': 1,
            'names': ['defect']
        }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False, default_flow_style=False)
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"Final dataset created at: {final_dataset_dir}")
    print(f"Training config file created at: {yaml_path}")

    # Clean up intermediate files
    shutil.rmtree(intermediate_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the coffee bean dataset for YOLO training.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=['single-class', 'multi-class'],
        help="The dataset configuration to prepare ('single-class' for all defects, 'multi-class' for specific types)."
    )
    args = parser.parse_args()
    main(args)