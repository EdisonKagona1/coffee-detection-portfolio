"""
A collection of utility functions for the Coffee Bean Detection project.

This module provides a robust toolkit for common tasks throughout the project,
including data visualization, coordinate system transformations, and file
manipulation. The functions here are designed to be reusable and are imported
by the main scripts (`visualize_data.py`, `prepare_dataset.py`, etc.).
"""
import os
import cv2
import json
import shutil
import fnmatch
import subprocess
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib import patches
from typing import List, Tuple, Any, TextIO, Dict

# Define a type alias for a bounding box for improved code clarity
BoundingBox = List[float]


def display_image_and_box(img: np.ndarray, bbox_file: TextIO, name: str, save: bool = False, display: bool = True) -> None:
    """
    Displays an image side-by-side with a version that has bounding boxes drawn on it.
    Reads bounding boxes from a YOLO-format .txt file.

    Args:
        img (np.ndarray): The image (in BGR format) to display.
        bbox_file (TextIO): An open file-like object containing YOLO format labels.
        name (str): The name of the image, used for the plot title and saving.
        save (bool): If True, saves the visualization to the 'outputs/label_verification' directory.
        display (bool): If True, shows the plot on screen.
    """
    height, width, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=160, figsize=(16, 8))
    fig.suptitle(f"Label Verification: {name}", fontsize=16)

    ax1.imshow(img_rgb)
    ax1.set_title("Original Image")
    ax1.axis('off')

    ax2.imshow(img_rgb)
    ax2.set_title("Image with Ground-Truth Boxes")
    ax2.axis('off')

    for bbox_str in bbox_file.readlines():
        try:
            _, center_x, center_y, width_x, width_y = map(float, bbox_str.split(" "))
            
            # Convert YOLO format (normalized, center-based) to pixel coordinates
            box_width = width_x * width
            box_height = width_y * height
            anchor_x = (center_x * width) - (box_width / 2)
            anchor_y = (center_y * height) - (box_height / 2)

            rect = patches.Rectangle(
                (anchor_x, anchor_y), box_width, box_height,
                linewidth=1.5, edgecolor='lime', facecolor='none'
            )
            ax2.add_patch(rect)
        except ValueError:
            print(f"Warning: Skipping malformed line in label file for {name}: '{bbox_str.strip()}'")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle

    if save:
        save_dir = os.path.join("outputs", "label_verification")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, name))
    
    if display:
        plt.show()
    
    plt.close(fig) # Important for running in loops to prevent memory leaks


def save_prediction_image(img_full: np.ndarray, prediction: Any, save_path: str, true_count: int = None) -> None:
    """
    Saves an image with predicted bounding boxes and optional counts drawn on it.

    Args:
        img_full (np.ndarray): The full image (in BGR format) to draw on.
        prediction (Any): The prediction object from a YOLO model.
        save_path (str): The full path, including filename, where the image will be saved.
        true_count (int, optional): The ground-truth count of objects for comparison.
    """
    all_boxes = prediction[0].boxes.cpu()
    esti_count = len(all_boxes)

    fig, ax = plt.subplots(figsize=(18.5, 10.5), dpi=100)

    rgb_img = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)
    ax.imshow(rgb_img)

    for box in all_boxes:
        xb, yb, xe, ye, _, _ = box.data[0]
        rect = patches.Rectangle(
            (xb, yb), xe - xb, ye - yb,
            linewidth=1, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)

    ax.axis('off')
    if true_count is not None:
        ax.set_title(f"Ground Truth Count: {true_count} | Estimated Count: {esti_count}", fontsize=16)
    else:
        ax.set_title(f"Estimated Count: {esti_count}", fontsize=16)
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def xml_to_yolo(annotations: List[Any], img_width: int, img_height: int) -> str:
    """
    Converts a list of XML annotations to a single string in YOLO format.

    Args:
        annotations (List[Any]): A list of annotation objects from an ElementTree parse.
        img_width (int): The width of the image for normalization.
        img_height (int): The height of the image for normalization.

    Returns:
        str: A multi-line string with one YOLO-formatted bounding box per line.
    """
    yolo_formatted_str = ""
    for annotation in annotations:
        bndbox = annotation.find("bndbox")
        xmin, ymin = int(bndbox.find("xmin").text), int(bndbox.find("ymin").text)
        xmax, ymax = int(bndbox.find("xmax").text), int(bndbox.find("ymax").text)

        dw, dh = 1.0 / img_width, 1.0 / img_height
        center_x, center_y = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        width_x, width_y = float(xmax - xmin), float(ymax - ymin)

        norm_center_x, norm_center_y = center_x * dw, center_y * dh
        norm_width_x, norm_width_y = width_x * dw, width_y * dh

        class_index = 0 # Assuming single-class problem for this direct conversion
        yolo_formatted_str += f"{class_index} {norm_center_x} {norm_center_y} {norm_width_x} {norm_width_y}\n"

    return yolo_formatted_str


def calculate_slice_bboxes(image_height: int, image_width: int, slice_height: int, slice_width: int,
                           overlap_height_ratio: float, overlap_width_ratio: float) -> List[List[int]]:
    """
    Calculates overlapping slice coordinates for a large image. This is a core
    component of the advanced preprocessing strategy.

    Args:
        image_height: Height of the original image.
        image_width: Width of the original image.
        slice_height: Height of each slice.
        slice_width: Width of each slice.
        overlap_height_ratio: Fractional overlap in height of each slice.
        overlap_width_ratio: Fractional overlap in width of each slice.

    Returns:
        A list of slice bounding boxes in [xmin, ymin, xmax, ymax] format.
    """
    slice_bboxes = []
    y_max = y_min = 0
    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)
    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes


def run_command(cmd: str, verbose: bool = False) -> None:
    """
    Runs a shell command using subprocess, capturing output.

    Args:
        cmd (str): The command to run as a single string.
        verbose (bool): If True, prints the command's stdout and stderr.
    """
    print(f"Executing command: {cmd}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
        encoding='utf-8',
        errors='ignore'
    )
    std_out, std_err = process.communicate()
    if verbose:
        if std_out:
            print(f"STDOUT: {std_out.strip()}")
        if std_err:
            print(f"STDERR: {std_err.strip()}")