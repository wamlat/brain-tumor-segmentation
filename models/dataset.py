import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np

class BraTSDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the subject folders.
            transform (callable, optional): Optional transform to apply to images/masks.
        """
        self.root_dir = root_dir
        self.subjects = os.listdir(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_id = self.subjects[idx]
        subject_path = os.path.join(self.root_dir, subject_id)

        # Load all modalities
        t1n = nib.load(os.path.join(subject_path, f'{subject_id}-t1n.nii.gz')).get_fdata()
        t1c = nib.load(os.path.join(subject_path, f'{subject_id}-t1c.nii.gz')).get_fdata()
        t2f = nib.load(os.path.join(subject_path, f'{subject_id}-t2f.nii.gz')).get_fdata()
        t2w = nib.load(os.path.join(subject_path, f'{subject_id}-t2w.nii.gz')).get_fdata()

        # Stack into channels (C, H, W)
        image = np.stack([t1n, t1c, t2f, t2w], axis=0)

        # Load the segmentation mask
        mask = nib.load(os.path.join(subject_path, f'{subject_id}-seg.nii.gz')).get_fdata()

        # Pick a random slice where there is tumor
        slice_idxs = np.unique(np.where(mask > 0)[2])  # find slices that have tumor
        if len(slice_idxs) == 0:
            slice_idx = image.shape[2] // 2  # fallback to middle slice
        else:
            slice_idx = np.random.choice(slice_idxs)

        image_slice = image[:, :, :, slice_idx]
        mask_slice = mask[:, :, slice_idx]

        # Normalize image intensities (important for MRIs)
        image_slice = (image_slice - np.mean(image_slice)) / (np.std(image_slice) + 1e-8)

        # Convert to tensor
        image_tensor = torch.tensor(image_slice, dtype=torch.float32)
        mask_tensor = torch.tensor(mask_slice, dtype=torch.float32).unsqueeze(0)  # add channel dim

        # Optional transforms
        if self.transform:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

        return image_tensor, mask_tensor
