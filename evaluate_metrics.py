import torch
import numpy as np
from models.unet import UNet
from models.dataset import BraTSDataset
from sklearn.metrics import precision_score, recall_score, f1_score
import os

# Settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = './outputs/checkpoints/best_model.pth'

# Load model
model = UNet(in_channels=4, out_channels=1)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()

# Load dataset
dataset = BraTSDataset(root_dir='./data_original/raw/GLI-Training')

# Pick number of samples
NUM_SAMPLES = min(200, len(dataset))  # You can make it bigger if you want

# Initialize lists
y_true = []
y_pred = []

START_INDEX = 1000
for idx in range(START_INDEX, START_INDEX + NUM_SAMPLES):
    image, true_mask = dataset[idx]
    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output)
        predicted_mask = (output > 0.5).float().squeeze().cpu().numpy()

    # Ground truth
    true_mask = true_mask.squeeze().cpu().numpy()

    # Binarize (some true masks might have float values)
    true_binary = (true_mask > 0).astype(int)
    pred_binary = (predicted_mask > 0).astype(int)

    # For this slice: was there any tumor pixel?
    true_tumor = 1 if true_binary.sum() > 0 else 0
    pred_tumor = 1 if pred_binary.sum() > 0 else 0

    y_true.append(true_tumor)
    y_pred.append(pred_tumor)

# Calculate metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n=== Quantitative Evaluation ===")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
