# 4_retrain.py
# ─────────────────────────────────────────────────────────
# This is the CONTINUOUS LEARNING script.
#
# When you have new labeled wound images (image + mask pairs),
# run this script to retrain the model and improve its accuracy.
#
# What it does differently from 3_train.py:
#   - Loads the EXISTING best model (doesn't start from scratch)
#   - Trains for fewer epochs (model already knows a lot)
#   - Combines old + new data automatically
#   - Saves updated model, replacing the old best if improved
#
# How to add new data:
#   1. Put new images in:  data/new_images/
#   2. Put new masks in:   data/new_masks/
#   3. Run: python 4_retrain.py
# ─────────────────────────────────────────────────────────

import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader, ConcatDataset

from config import (
    BEST_MODEL_PATH, MODEL_DIR, LOG_DIR,
    RETRAIN_EPOCHS, LEARNING_RATE, BATCH_SIZE, IMAGE_SIZE, RANDOM_SEED
)
from prepare_data import (
    build_dataloaders, find_image_mask_pairs,
    WoundDataset, get_train_transforms, get_val_transforms
)
from train import build_model, CombinedLoss, train_one_epoch, validate, dice_score, device


# ─────────────────────────────────────────────────────────
# New data directories
# ─────────────────────────────────────────────────────────

NEW_IMAGE_DIR = "data/new_images"    # drop new images here
NEW_MASK_DIR  = "data/new_masks"     # drop matching masks here


# ─────────────────────────────────────────────────────────
# Load the existing best model
# ─────────────────────────────────────────────────────────

def load_existing_model():
    """
    Load the best model saved from a previous training run.
    This is called 'transfer learning' or 'fine-tuning' —
    we're not starting from zero, we're starting from what
    the model already knows.
    """
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {BEST_MODEL_PATH}\n"
            "Please run python 3_train.py first."
        )

    model = build_model()
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    previous_dice = checkpoint.get("best_dice", 0.0)
    previous_epoch = checkpoint.get("epoch", 0)

    print(f"  ✓ Loaded model from epoch {previous_epoch}")
    print(f"  ✓ Previous best Dice: {previous_dice:.4f}")

    return model, previous_dice


# ─────────────────────────────────────────────────────────
# Combine old + new data
# ─────────────────────────────────────────────────────────

def build_combined_dataloaders():
    """
    Combine the original dataset with new labeled images.
    The model trains on everything together.
    """
    print("\n[Data] Loading original dataset...")
    train_loader, val_loader, _, _ = build_dataloaders()

    # Check if new data exists
    has_new_data = (
        os.path.exists(NEW_IMAGE_DIR) and
        os.path.exists(NEW_MASK_DIR) and
        len(os.listdir(NEW_IMAGE_DIR)) > 0
    )

    if has_new_data:
        print("\n[Data] Loading new images...")
        new_pairs = find_image_mask_pairs(NEW_IMAGE_DIR, NEW_MASK_DIR)

        if new_pairs:
            new_dataset = WoundDataset(
                new_pairs,
                transform=get_train_transforms(IMAGE_SIZE)
            )

            # Combine original training data + new data
            combined_dataset = ConcatDataset([
                train_loader.dataset,
                new_dataset
            ])
            train_loader = DataLoader(
                combined_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )
            print(f"  ✓ Combined dataset: {len(combined_dataset)} images total")
            print(f"    ({len(new_pairs)} new images added)")
    else:
        print(f"\n  ℹ No new data found in {NEW_IMAGE_DIR}")
        print("    Retraining on original data only (still useful for more epochs)")

    return train_loader, val_loader


# ─────────────────────────────────────────────────────────
# MAIN: Run retraining
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)
    os.makedirs(NEW_IMAGE_DIR, exist_ok=True)
    os.makedirs(NEW_MASK_DIR,  exist_ok=True)

    print("=" * 50)
    print("  Wound Model — Retraining with New Data")
    print("=" * 50)

    # Load existing model
    print("\n[1/3] Loading existing model...")
    model, previous_best_dice = load_existing_model()

    # Load combined data
    print("\n[2/3] Preparing data...")
    train_loader, val_loader = build_combined_dataloaders()

    # Set up training with a lower learning rate
    # (lower LR = smaller adjustments, so we don't overwrite what model already knows)
    loss_fn   = CombinedLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE * 0.1,    # 10x smaller than initial training
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=RETRAIN_EPOCHS
    )

    history  = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
    best_dice = previous_best_dice

    print(f"\n[3/3] Retraining for {RETRAIN_EPOCHS} epochs...")
    print("─" * 50)

    for epoch in range(1, RETRAIN_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{RETRAIN_EPOCHS}")

        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss,   val_dice   = validate(model, val_loader, loss_fn)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)

        print(f"  Train — Loss: {train_loss:.4f} | Dice: {train_dice:.4f}")
        print(f"  Val   — Loss: {val_loss:.4f}   | Dice: {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            # Save with a timestamp so you keep a history of improvements
            timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path  = os.path.join(MODEL_DIR, f"best_model_retrained_{timestamp}.pth")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "best_dice":   best_dice,
            }, save_path)
            # Also overwrite the main best model
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "best_dice":   best_dice,
            }, BEST_MODEL_PATH)
            improvement = best_dice - previous_best_dice
            print(f"  ✓ New best model saved! Dice improved by +{improvement:.4f}")

    print(f"""
─────────────────────────────────────
Retraining complete!
  Previous Dice: {previous_best_dice:.4f}
  New best Dice: {best_dice:.4f}
  Improvement:   +{best_dice - previous_best_dice:.4f}

Tip: move processed new images from data/new_images/
     to data/images/ so they're included in future
     retraining runs automatically.
─────────────────────────────────────
""")
