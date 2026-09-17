# 3_train.py
# ─────────────────────────────────────────────────────────
# This is the main training script. It:
#  1. Builds a U-Net model (a powerful image segmentation network)
#  2. Trains it on your wound dataset
#  3. Evaluates on validation set after each epoch
#  4. Saves the best model automatically
#  5. Shows live training progress
# ─────────────────────────────────────────────────────────

import os
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (
    BEST_MODEL_PATH, LAST_MODEL_PATH, MODEL_DIR,
    LOG_DIR, NUM_EPOCHS, LEARNING_RATE, NUM_CLASSES
)
from prepare_data import build_dataloaders


# ─────────────────────────────────────────────────────────
# STEP 1: Set up device (GPU or CPU)
# ─────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[Device] Using: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


# ─────────────────────────────────────────────────────────
# STEP 2: Build the U-Net model
# ─────────────────────────────────────────────────────────
# U-Net is a neural network shaped like the letter U:
#   - Left side (encoder): shrinks the image to extract features
#   - Bottom: smallest representation of the image
#   - Right side (decoder): grows back to original size with predictions
#
# We use ResNet34 as the encoder backbone — a proven feature extractor
# pre-trained on millions of images (ImageNet), so it already knows
# edges, textures, shapes. We just need to teach it wound-specific patterns.

def build_model():
    model = smp.Unet(
        encoder_name    = "resnet34",       # backbone (feature extractor)
        encoder_weights = "imagenet",       # start from pre-trained weights
                                            # (much faster than random start)
        in_channels     = 3,               # RGB images (3 colour channels)
        classes         = NUM_CLASSES,     # 1 output class (wound / not wound)
        activation      = None,            # we apply sigmoid separately
    )
    return model.to(device)


# ─────────────────────────────────────────────────────────
# STEP 3: Loss functions
# ─────────────────────────────────────────────────────────
# Loss = how wrong the model's prediction is.
# The model tries to minimise this number during training.
#
# We combine two losses:
#   BCEWithLogitsLoss: standard binary classification loss
#   DiceLoss:          specifically good for segmentation —
#                      penalises missing the wound area heavily

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce        = nn.BCEWithLogitsLoss()
        self.dice_loss  = smp.losses.DiceLoss(mode="binary")
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight

    def forward(self, predictions, targets):
        bce_loss  = self.bce(predictions, targets)
        dice_loss = self.dice_loss(predictions, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ─────────────────────────────────────────────────────────
# STEP 4: Metric — Dice Score (how accurate is the model?)
# ─────────────────────────────────────────────────────────
# Dice Score = 2 × (overlap between prediction and ground truth)
#                   / (total pixels in prediction + ground truth)
#
# Score of 1.0 = perfect prediction
# Score of 0.0 = completely wrong
# A good wound segmentation model achieves 0.80+    

def dice_score(predictions, targets, threshold=0.5):
    # Apply sigmoid to convert raw outputs → probabilities (0 to 1)
    preds  = torch.sigmoid(predictions) > threshold
    preds  = preds.float()

    intersection = (preds * targets).sum()
    dice         = (2.0 * intersection) / (preds.sum() + targets.sum() + 1e-8)
    return dice.item()


# ─────────────────────────────────────────────────────────
# STEP 5: Training loop (one epoch)
# ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn):
    """Train for one full pass through the training data."""
    model.train()  # tell the model it's in training mode
    total_loss  = 0
    total_dice  = 0

    # tqdm shows a progress bar
    progress = tqdm(loader, desc="  Training", leave=False)

    for images, masks in progress:
        # Move data to GPU
        images = images.to(device)
        masks  = masks.to(device)

        # ── Forward pass ──────────────────────────────────
        # Pass images through the model to get predictions
        predictions = model(images)

        # ── Calculate loss ────────────────────────────────
        # How wrong is the model right now?
        loss = loss_fn(predictions, masks)

        # ── Backward pass ─────────────────────────────────
        # Calculate gradients (which direction to adjust weights)
        optimizer.zero_grad()   # clear old gradients
        loss.backward()         # calculate new gradients
        optimizer.step()        # update model weights

        # Track metrics
        total_loss += loss.item()
        total_dice += dice_score(predictions, masks)

        progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    return avg_loss, avg_dice


# ─────────────────────────────────────────────────────────
# STEP 6: Validation loop (check performance on unseen data)
# ─────────────────────────────────────────────────────────

def validate(model, loader, loss_fn):
    """Evaluate model on validation set (no weight updates)."""
    model.eval()  # tell the model it's in evaluation mode
    total_loss = 0
    total_dice = 0

    with torch.no_grad():   # don't calculate gradients (saves memory)
        for images, masks in tqdm(loader, desc="  Validating", leave=False):
            images      = images.to(device)
            masks       = masks.to(device)
            predictions = model(images)
            loss        = loss_fn(predictions, masks)

            total_loss += loss.item()
            total_dice += dice_score(predictions, masks)

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    return avg_loss, avg_dice


# ─────────────────────────────────────────────────────────
# STEP 7: Plot and save training history
# ─────────────────────────────────────────────────────────

def plot_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train loss")
    ax1.plot(history["val_loss"],   label="Val loss")
    ax1.set_title("Loss over epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["train_dice"], label="Train Dice")
    ax2.plot(history["val_dice"],   label="Val Dice")
    ax2.set_title("Dice Score over epochs (higher = better)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Chart saved to: {save_path}")


# ─────────────────────────────────────────────────────────
# MAIN: Run the full training
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    # Load data
    print("\n[1/4] Loading data...")
    train_loader, val_loader, test_loader, _ = build_dataloaders()

    # Build model
    print("\n[2/4] Building U-Net model...")
    model    = build_model()
    loss_fn  = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Learning rate scheduler — reduces LR if model stops improving
    # This helps squeeze out extra performance in later epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    # Training history (for plotting)
    history = {
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": []
    }

    best_dice      = 0.0
    epochs_no_improve = 0
    EARLY_STOP_PATIENCE = 10  # stop if no improvement for 10 epochs

    print(f"\n[3/4] Training for {NUM_EPOCHS} epochs...")
    print("─" * 55)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss,   val_dice   = validate(model, val_loader, loss_fn)

        # Update learning rate based on validation Dice
        scheduler.step(val_dice)

        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)

        print(f"  Train — Loss: {train_loss:.4f} | Dice: {train_dice:.4f}")
        print(f"  Val   — Loss: {val_loss:.4f}   | Dice: {val_dice:.4f}")

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_dice":       best_dice,
            }, BEST_MODEL_PATH)
            print(f"  ✓ Best model saved (Dice: {best_dice:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping — prevents overfitting
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping — no improvement for {EARLY_STOP_PATIENCE} epochs")
            break

    # Save last model (for retraining later)
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "best_dice":   best_dice,
    }, LAST_MODEL_PATH)

    # Plot training history
    print("\n[4/4] Saving training chart...")
    plot_history(history, os.path.join(LOG_DIR, "training_history.png"))

    print(f"""
─────────────────────────────────────
Training complete!
  Best Dice Score: {best_dice:.4f}
  Best model:      {BEST_MODEL_PATH}
  Training chart:  logs/training_history.png

Next: python 5_predict.py to test on a new image
─────────────────────────────────────
""")
