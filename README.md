# Brain Tumor Segmentation (BraTS Dataset)

This project trains and evaluates a UNet-based model to segment brain tumors from MRI images.

## Project Structure
- `models/`: UNet architecture, dataset loading, and utility functions
- `notebooks/`: Jupyter notebooks for data exploration and model training
- `evaluate.py`: Overlay predicted tumor masks on MRI slices
- `evaluate_failures.py`: Identify and visualize model failure cases
- `evaluate_metrics.py`: Quantitative evaluation (precision, recall, F1)

## How to Run
1. Train the UNet model (`TrainUNet.ipynb`) and save the checkpoint to `./outputs/checkpoints/best_model.pth`.
2. Evaluate results:
   - Run `evaluate.py` for overlays
   - Run `evaluate_failures.py` for error maps and worst-case analysis
   - Run `evaluate_metrics.py` for precision/recall/F1

## Requirements
Install packages with:

```bash
pip install -r requirements.txt
