# AI-Powered Defect Detection and Active Quality Assurance

This repository documents an end-to-end deep learning project that evolves from a standard object detector into a sophisticated, cognitive AI system. It demonstrates a complete, hands-on workflow from processing raw, non-standard data to implementing advanced concepts like two-stage modeling and active inference.

The project showcases in-depth, practical experience in developing and adapting machine learning algorithms using PyTorch and its ecosystem, directly addressing topics relevant to modern AI research.

---

### Key Features & Skills Demonstrated

*   **Hands-on Algorithm Adaptation:** Custom algorithms for data parsing (XML to YOLO), image processing (overlapping tiling), and programmatic label remapping.
*   **PyTorch-based Model Training:** Systematic experimentation and training of YOLOv8 models.
*   **Advanced Quantitative Validation:**
    *   **Two-Stage Modeling:** A deep learning detector followed by a classical polynomial regression model to correct for systematic biases (e.g., underestimation due to occlusion).
    *   **Uncertainty Estimation:** Implementation of Monte Carlo (MC) Dropout to approximate Bayesian inference and quantify model uncertainty.
*   **Active Inference for Shared Autonomy:** A simulated system where the AI agent uses its uncertainty to intelligently decide when to "query" a human expert, optimizing the human-in-the-loop workflow.
*   **Professional Project Structure:** A reproducible workflow with dedicated, documented scripts for each stage of the pipeline.

---

## The Project Workflow

This project is structured in two main parts: a **Core Detection Pipeline** and an **Active Inference Extension**.

### Part 1: Core Detection Pipeline

The foundational object detection system. The exploratory work originally done in Jupyter Notebooks has been refactored into clean, reusable command-line scripts.

1.  **`prepare_dataset.py`:** Transforms the raw, messy dataset into a clean, trainable format.
2.  **`train.py`:** Runs the training loop with configurable hyperparameters.
3.  **`inference.py`:** Uses a trained model for simple prediction on new data.
4.  **`validate.py`:** Rigorously evaluates performance and applies the two-stage statistical correction model.

### Part 2: Active Inference Extension

This extension transforms the passive detector into an intelligent agent, demonstrating concepts directly relevant to the "Model-based shared autonomy via active inference" research topic.

5.  **`mc_inference.py`:** Estimates model uncertainty using the Monte Carlo Dropout technique.
6.  **`active_validator.py`:** Simulates a shared autonomy loop where the AI uses its uncertainty to decide when to ask a human for help, minimizing expert effort while maximizing accuracy.

---

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/EdisonKagona1/coffee-detection-portfolio.git
    cd coffee-detection-portfolio
    ```

2.  Create and activate a virtual environment (recommended):
    ```bash
    # For Conda (recommended)
    conda create -n coffee_env python=3.10 -y
    conda activate coffee_env
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage & Demonstration

**Note:** This repository contains the complete codebase but does not include the proprietary image dataset due to contractual agreements. The scripts are fully functional and can be run on a similarly structured dataset. Placeholder/dummy files can be used to test the execution flow of each script.

### 1. Prepare the Dataset

This script handles all preprocessing, including XML conversion and image slicing. It expects a `data/raw/` directory.

```bash
# Prepare the multi-class dataset for training
python prepare_dataset.py --config multi-class

# Train the multi-class model ---  Once the data is prepared, train.py runs the training process.
python train.py --data data/configs/coffee_multi-class.yaml --epochs 80 --name coffee_multi_class_final

# Validate the model and apply the error correction model --- Use validate.py to assess the model and apply the two-stage polynomial correction.
python validate.py --model-path runs/train/coffee_multi_class_final/weights/best.pt --csv-path data/validation_counts.csv --images-dir data/validation_images

# Run the active inference simulation -- Simulate the shared autonomy workflow with active_validator.py.
python active_validator.py --weights runs/train/coffee_multi_class_final/weights/best.pt --csv-path data/validation_counts.csv --images-dir data/validation_images



"""   
Technical Highlights
- A core part of this project was developing custom algorithms to solve real-world challenges.
- Custom Data Ingestion Pipeline (prepare_dataset.py): An engineering solution was developed to handle a complex dataset, featuring a custom XML parser, an advanced overlapping tiling algorithm for high-resolution images, and programmatic label remapping for controlled experimentation.
- Two-Stage Predictive Modeling (validate.py): Analysis revealed that the base YOLO model systematically underestimated counts due to object occlusion. To solve this, a two-stage model was implemented where a Scikit-learn polynomial regression model learns to correct the output of the deep learning detector, significantly improving the practical accuracy.

- Approximating Bayesian Inference (mc_inference.py): To enable the model to reason about its own knowledge, Monte Carlo Dropout was implemented. This technique provides a practical way to estimate model uncertainty for each prediction, a prerequisite for any probabilistic or active inference system.

- Active Inference Loop (active_validator.py): The project culminates in a simulation of an intelligent agent. The system uses its calculated uncertainty to make an economic decision: if its uncertainty is high, it "queries" a human expert; otherwise, it "commits" to its own prediction. This demonstrates a core principle of active inference—acting to minimize future surprise—and is a direct implementation of a shared autonomy concept.
 """