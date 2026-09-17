# 2_prepare_data.py
# ─────────────────────────────────────────────────────────
# This script:
#  1. Loads all images + masks from your data folder
#  2. Checks that every image has a matching mask
#  3. Cleans images (resize, normalise)
#  4. Splits into train / validation / test sets
#  5. Creates a PyTorch Dataset class used during training
# ─────────────────────────────────────────────────────────

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import (IMAGE_DIR, MASK_DIR, IMAGE_SIZE,
                    BATCH_SIZE, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED)


# ─────────────────────────────────────────────────────────
# STEP 1: Find and validate image-mask pairs
# ─────────────────────────────────────────────────────────

def find_image_mask_pairs(image_dir, mask_dir):
    """
    Find all images. If a mask exists, pair it.
    If no mask exists, pair it with None (which tells the Dataset to make a blank mask).
    """
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    mask_files_set = set(os.listdir(mask_dir))

    pairs = []
    blank_mask_count = 0

    for img_file in image_files:
        if img_file in mask_files_set:
            # Mask exists
            pairs.append((
                os.path.join(image_dir, img_file),
                os.path.join(mask_dir, img_file)
            ))
        else:
            # No mask exists -> use None as a flag
            pairs.append((
                os.path.join(image_dir, img_file),
                None  
            ))
            blank_mask_count += 1
            
    print(f"  Found {len(pairs)} total images")
    print(f"    - {len(pairs) - blank_mask_count} have real masks")
    print(f"    - {blank_mask_count} will use generated blank masks (normal images)")

    return pairs


# ─────────────────────────────────────────────────────────
# STEP 2: Define image transforms (cleaning + augmentation)
# ─────────────────────────────────────────────────────────

def get_train_transforms(image_size):
    """
    Transforms applied to TRAINING images only.
    Augmentation = artificially creating variations of your images
    so the model learns to handle different real-world conditions.
    """
    return A.Compose([
        # Resize all images to the same size
        A.Resize(image_size[0], image_size[1]),

        # Randomly flip horizontally (a wound on the left looks like
        # one on the right — both are valid training examples)
        A.HorizontalFlip(p=0.5),

        # Randomly rotate up to 30 degrees
        # (wounds can appear at any angle in a photo)
        A.Rotate(limit=30, p=0.5),

        # Randomly adjust brightness and contrast
        # (accounts for different lighting conditions)
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),

        # Slightly blur sometimes (mimics out-of-focus photos)
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),

        # Normalise pixel values to standard range
        # (helps model training converge faster)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet standard mean
            std=[0.229, 0.224, 0.225]     # ImageNet standard std
        ),

        # Convert numpy array → PyTorch tensor
        ToTensorV2()
    ])


def get_val_transforms(image_size):
    """
    Transforms for VALIDATION and TEST images.
    No augmentation — we want to measure real performance.
    """
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


# ─────────────────────────────────────────────────────────
# STEP 3: PyTorch Dataset class
# ─────────────────────────────────────────────────────────

class WoundDataset(Dataset):
    """
    A PyTorch Dataset loads images one by one during training.
    Think of it as a smart list that:
      - reads an image from disk
      - reads its mask
      - applies transforms
      - returns them as tensors the model can read
    """

    def __init__(self, pairs, transform=None):
        """
        pairs     = list of (image_path, mask_path) tuples
        transform = cleaning/augmentation pipeline to apply
        """
        self.pairs     = pairs
        self.transform = transform

    def __len__(self):
        # PyTorch calls this to know how many items are in the dataset
        return len(self.pairs)

    def __getitem__(self, idx):
        # PyTorch calls this to get item number `idx`
        image_path, mask_path = self.pairs[idx]

        # 1. Load image in colour (BGR → RGB)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Load OR Generate Mask
        if mask_path is not None:
            # Load real mask in grayscale
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Force mask dimensions to match image dimensions
            if image.shape[:2] != mask.shape[:2]:
                mask = cv2.resize(
                    mask, 
                    (image.shape[1], image.shape[0]), 
                    interpolation=cv2.INTER_NEAREST
                )
        else:
            # Generate a blank (all black/zero) mask of the exact same size as the image
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # 3. Convert mask to binary: 0 = background, 1 = wound
        mask = (mask > 127).astype(np.float32)

        # 4. Apply transforms if provided
        if self.transform:
            result = self.transform(image=image, mask=mask)
            image  = result["image"]
            mask   = result["mask"].unsqueeze(0)   # add channel dimension

        return image, mask


# ─────────────────────────────────────────────────────────
# STEP 4: Build DataLoaders (what the training loop uses)
# ─────────────────────────────────────────────────────────

def build_dataloaders(image_dir=IMAGE_DIR, mask_dir=MASK_DIR):
    """
    Returns three DataLoaders: train, validation, test.
    DataLoader = automatically batches and shuffles data during training.
    """
    print("\n[Data Preparation]")

    # 1. Find pairs
    all_pairs = find_image_mask_pairs(image_dir, mask_dir)
    if len(all_pairs) == 0:
        raise RuntimeError(
            "No image-mask pairs found!\n"
            "Make sure your images are in data/images/ and\n"
            "masks are in data/masks/ with matching filenames."
        )

    # 2. Split into train / val / test
    #    e.g. 1000 images → 750 train, 150 val, 100 test
    train_pairs, temp_pairs = train_test_split(
        all_pairs,
        test_size=(VAL_SPLIT + TEST_SPLIT),
        random_state=RANDOM_SEED
    )
    val_pairs, test_pairs = train_test_split(
        temp_pairs,
        test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT),
        random_state=RANDOM_SEED
    )

    print(f"  Train:      {len(train_pairs)} images")
    print(f"  Validation: {len(val_pairs)} images")
    print(f"  Test:       {len(test_pairs)} images")

    # 3. Create Dataset objects
    train_dataset = WoundDataset(train_pairs, transform=get_train_transforms(IMAGE_SIZE))
    val_dataset   = WoundDataset(val_pairs,   transform=get_val_transforms(IMAGE_SIZE))
    test_dataset  = WoundDataset(test_pairs,  transform=get_val_transforms(IMAGE_SIZE))

    # 4. Wrap in DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,          # shuffle order every epoch
        num_workers=2,         # load images in parallel
        pin_memory=True        # faster GPU transfer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,          # test one at a time for detailed results
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader, test_loader, train_pairs


# ─────────────────────────────────────────────────────────
# Run this file directly to test your dataset
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_loader, val_loader, test_loader, _ = build_dataloaders()

    # Show a sample batch to confirm everything works
    images, masks = next(iter(train_loader))
    print(f"\n  Sample batch — image shape: {images.shape}")
    print(f"  Sample batch — mask shape:  {masks.shape}")
    print(f"\n  ✓ Data preparation complete! Run python train.py next.")
