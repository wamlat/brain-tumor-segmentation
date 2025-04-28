import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from models.unet import UNet
from models.dataset import BraTSDataset

# ================== Settings ==================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = './outputs/checkpoints/best_model.pth'
DATASET_PATH = './data_original/raw/GLI-Training'
START_INDEX = 1000
NUM_SAMPLES = 100

COMPARISON_DIR = './outputs/eval_comparisons'
FAILURE_DIR = './outputs/failure_cases'
ERROR_MAP_DIR = './outputs/eval_error_maps'
TOP10_DIR = './outputs/eval_top10_worst'

for path in [COMPARISON_DIR, FAILURE_DIR, ERROR_MAP_DIR, TOP10_DIR]:
    os.makedirs(path, exist_ok=True)


# ================== Utilities ==================

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def plot_basic_comparison(mri, true_mask, pred_mask, save_path, title_suffix=""):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(mri, cmap='gray')
    axs[0].set_title('MRI')
    axs[0].axis('off')

    axs[1].imshow(true_mask, cmap='Reds')
    axs[1].set_title('Ground Truth Mask')
    axs[1].axis('off')

    axs[2].imshow(pred_mask, cmap='Blues')
    axs[2].set_title('Predicted Mask' + title_suffix)
    axs[2].axis('off')

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_error_map(mri, true_mask, pred_mask, save_path):
    missed_tumor = np.logical_and(true_mask > 0.5, pred_mask <= 0)
    # hopefully fixes the blue missed not showing up issue
    false_alarm = np.logical_and(true_mask <= 0.5, pred_mask > 0.5)

    error_map = np.zeros_like(mri)
    error_map[missed_tumor] = -1  # Red
    error_map[false_alarm] = 1  # Blue

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    axs[0].imshow(mri, cmap='gray')
    axs[0].set_title('MRI')
    axs[0].axis('off')

    axs[1].imshow(true_mask, cmap='Reds')
    axs[1].set_title('Ground Truth Mask')
    axs[1].axis('off')

    axs[2].imshow(pred_mask, cmap='Blues')
    axs[2].set_title('Predicted Mask')
    axs[2].axis('off')

    axs[3].imshow(mri, cmap='gray', alpha=0.8)
    axs[3].imshow(error_map, cmap='bwr', alpha=0.7, vmin=-1, vmax=1)
    axs[3].set_title('Error Map\n(Blue: Missed, Red: False Positive)')
    axs[3].axis('off')

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


# ================== Main ==================

def main():
    # Load model
    model = UNet(in_channels=4, out_channels=1)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    # Load dataset
    dataset = BraTSDataset(root_dir=DATASET_PATH)

    failure_count = 0
    iou_records = []

    # Evaluate
    for idx in range(START_INDEX, START_INDEX + NUM_SAMPLES):
        image, true_mask = dataset[idx]
        image = image.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(image)
            output = torch.sigmoid(output)
            pred_mask = (output > 0.5).float().squeeze().cpu().numpy()

        true_mask = true_mask.squeeze().numpy()
        mri_slice = image.squeeze()[0].cpu().numpy()

        # Compute IoU
        iou = compute_iou(true_mask, pred_mask)
        iou_records.append((idx, iou))

        # Save general comparison
        comparison_path = os.path.join(COMPARISON_DIR, f'sample_{idx}.png')
        plot_basic_comparison(mri_slice, true_mask, pred_mask, comparison_path)

        # If failure, save more detailed error map
        if iou < 0.6:
            failure_count += 1

            failure_basic_path = os.path.join(FAILURE_DIR, f'failure_{idx}.png')
            plot_basic_comparison(mri_slice, true_mask, pred_mask, failure_basic_path)

            error_map_path = os.path.join(ERROR_MAP_DIR, f'error_map_{idx}.png')
            plot_error_map(mri_slice, true_mask, pred_mask, error_map_path)

    # Summarize
    print("\n=== Evaluation Completed ===")
    print(f"Samples Evaluated: {NUM_SAMPLES}")
    print(f"Failures (IoU < 0.6): {failure_count} ({failure_count / NUM_SAMPLES * 100:.2f}%)")

    # Sort and save Top-10 worst cases
    iou_records.sort(key=lambda x: x[1])
    print("\n=== Top 10 Worst Samples by IoU ===")
    for rank, (idx, iou) in enumerate(iou_records[:10], start=1):
        print(f"Rank {rank}: Sample {idx}, IoU = {iou:.4f}")

        src_path = os.path.join(ERROR_MAP_DIR, f'error_map_{idx}.png')
        dst_path = os.path.join(TOP10_DIR, f'top{rank:02d}_error_map_{idx}_iou{iou:.3f}.png')

        if os.path.exists(src_path):
            shutil.copyfile(src_path, dst_path)
            print(f"Copied {src_path} --> {dst_path}")
        else:
            print(f"Warning: {src_path} not found!")


if __name__ == '__main__':
    main()
