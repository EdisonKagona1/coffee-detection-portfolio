
# AI-Powered Defect Detection in Green Coffee Beans

This project presents an end-to-end deep learning pipeline for detecting defects in green coffee beans. It demonstrates a complete, hands-on workflow from processing raw, non-standard data to training, and finally performing a sophisticated two-stage quantitative analysis of the model's performance.

The entire pipeline was developed to overcome real-world data challenges and serve as a robust, automated quality control system. This repository showcases in-depth experience in adapting and developing machine learning algorithms using PyTorch and its ecosystem.

---

### Key Features & Skills Demonstrated

  **Hands-on Algorithm Adaptation:** Custom algorithms for data parsing (XML to YOLO), image processing (overlapping tiling), and programmatic label remapping.
  **PyTorch-based Model Training:** Systematic experimentation and training of YOLOv8 models.
  **Two-Stage Modeling:** A deep learning detector followed by a classical polynomial regression model to correct for systematic biases (e.g., underestimation due to occlusion).
  **End-to-End Project Structure:** A professional, reproducible workflow with dedicated scripts for each stage of the pipeline (data prep, training, validation).

---

## The Workflow

This project is structured as a professional, reproducible machine learning pipeline. The exploratory work originally done in Jupyter Notebooks has been refactored into a series of clean, documented, and reusable command-line scripts.

1.  **`prepare_dataset.py`:** Transforms the raw, messy dataset (images and XML labels) into a clean, trainable format.
2.  **`train.py`:** Runs the training loop with configurable hyperparameters.
3.  **`inference.py`:** Uses a trained model to make predictions on new data for visual inspection.
4.  **`validate.py`:** Rigorously evaluates and compares model performance on a practical counting task, including training and applying a statistical correction model.

---

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/EdisonKagona1/coffee-detection-portfolio.git
    cd coffee-detection-portfolio
    ```

2.  Create and activate a virtual environment (recommended):
    ```bash
    # For standard Python venv
    python -m venv venv
    source venv/Scripts/activate

    # For Conda
    conda create -n coffee_env python=3.10 -y
    conda activate coffee_env
    ```

3.  Install the required dependencies from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage Pipeline

**Note:** This repository contains the complete codebase but does not include the proprietary image dataset due to contractual agreements from the original project. The scripts are fully functional and can be run on a similarly structured dataset.

### 1. Prepare the Dataset

Place your raw images in `data/raw/images/`, XML labels in `data/raw/labels_xml/`, and a `notes.json` file defining categories in `data/raw/`. Then run:

```bash
# Prepare the multi-class dataset for training
python prepare_dataset.py --config multi-class

# Train the multi-class model for 80 epochs
python train.py --data data/configs/coffee_multi-class.yaml --epochs 80 --name coffee_multi_class_final

# Validate the model and apply the error correction model
python validate.py --model-path runs/train/coffee_multi_class_final/weights/best.pt --csv-path data/validation_counts.csv --images-dir data/validation_images

