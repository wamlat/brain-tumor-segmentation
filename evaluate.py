import torch
import matplotlib.pyplot as plt
from models.unet import UNet
from models.dataset import BraTSDataset
import numpy as np
import os
import cv2

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
NUM_SAMPLES = min(100, len(dataset))  # Safety cap

# Create output directory
os.makedirs('./outputs/eval_overlays', exist_ok=True)

# Store results
results = []

START_INDEX = 1000
for idx in range(START_INDEX, START_INDEX + NUM_SAMPLES):
    image, true_mask = dataset[idx]
    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output)
        predicted_mask = (output > 0.5).float().squeeze().cpu().numpy()

    # Find contours in predicted mask
    contours, _ = cv2.findContours(predicted_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Track detection result
    tumor_detected = len(contours) > 0
    results.append((idx, tumor_detected))

    # Get base MRI slice
    mri_slice = image.squeeze()[0].cpu().numpy()

    # Plot
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(mri_slice, cmap='gray', alpha=0.5)  # Lighten MRI background

    # Draw contours
    for contour in contours:
        contour = contour.squeeze()
        if contour.ndim == 2:
            ax.plot(contour[:, 0], contour[:, 1], color='red', linewidth=2)

    ax.set_title(f'MRI + Predicted Tumor Contour (Sample {idx})')
    ax.axis('off')

    # Save figure
    subject_id = f"patient{idx:05d}"
    slice_number = idx
    save_filename = f"{subject_id}_slice{slice_number:03d}_overlay.png"
    save_path = os.path.join('./outputs/eval_overlays', save_filename)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    if tumor_detected:
        print(f"Sample {idx}: Tumor detected!")
    else:
        print(f"Sample {idx}: No tumor detected.")

    print(f"Saved overlay to {save_path}")

# --- After all samples are processed, plot tumor detection heatmap ---

# Extract data
slice_indices = [item[0] for item in results]
detections = [item[1] for item in results]

# Color map: red = tumor, green = no tumor
colors = ['red' if det else 'green' for det in detections]

# Plot heatmap
plt.figure(figsize=(12, 1))
plt.scatter(slice_indices, [1]*len(slice_indices), c=colors, s=100)
plt.yticks([])
plt.title('Tumor Detection Across Samples')
plt.xlabel('Sample Index')

# Save heatmap
heatmap_path = './outputs/eval_overlays/detection_heatmap.png'
plt.savefig(heatmap_path, bbox_inches='tight')
plt.show()

print(f"\nSaved detection heatmap to {heatmap_path}")
