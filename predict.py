# 5_predict.py
# ─────────────────────────────────────────────────────────
# Run this to analyse a single image OR a full folder of images.
#
# To test a folder WITHOUT masks (just save predictions):
#   python 5_predict.py --image_folder data/test_images
#
# To test a folder WITH masks (calculates accuracy/Dice score):
#   python 5_predict.py --image_folder data/test_images --mask_folder data/test_masks
# ─────────────────────────────────────────────────────────

import os
import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import BEST_MODEL_PATH, IMAGE_SIZE, PIXELS_PER_CM
from train import build_model, device


# ─────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────

def load_model():
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {BEST_MODEL_PATH}\n"
            "Please run python 3_train.py first."
        )
    model      = build_model()
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"✓ Model loaded (Training Best Dice: {checkpoint.get('best_dice', '?'):.4f})")
    return model


# ─────────────────────────────────────────────────────────
# Pre-process a single image for inference
# ─────────────────────────────────────────────────────────

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    original = image.copy()
    image    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    orig_h, orig_w = image.shape[:2]
    resized = cv2.resize(image, (IMAGE_SIZE[1], IMAGE_SIZE[0]))

    mean   = np.array([0.485, 0.456, 0.406])
    std    = np.array([0.229, 0.224, 0.225])
    resized = (resized / 255.0 - mean) / std

    tensor = torch.from_numpy(resized).float()
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    return tensor, original, orig_h, orig_w


# ─────────────────────────────────────────────────────────
# Run prediction
# ─────────────────────────────────────────────────────────

def predict_mask(model, image_tensor, threshold=0.5):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output       = model(image_tensor)
        probability  = torch.sigmoid(output)
        mask         = (probability > threshold).float()

    mask = mask.squeeze().cpu().numpy()
    return mask, probability.squeeze().cpu().numpy()


# ─────────────────────────────────────────────────────────
# Measure wound dimensions
# ─────────────────────────────────────────────────────────

def measure_wound(mask, orig_h, orig_w, pixels_per_cm=PIXELS_PER_CM):
    mask_resized = cv2.resize(
        mask.astype(np.uint8),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST
    )

    contours, _ = cv2.findContours(
        mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None, None, mask_resized

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h    = cv2.boundingRect(largest)
    wound_px_w     = w
    wound_px_h     = h
    area_pixels    = cv2.contourArea(largest)

    if pixels_per_cm and pixels_per_cm > 0:
        length_cm  = max(wound_px_w, wound_px_h) / pixels_per_cm
        width_cm   = min(wound_px_w, wound_px_h) / pixels_per_cm
        area_cm2   = area_pixels / (pixels_per_cm ** 2)
        return length_cm, width_cm, area_cm2, mask_resized
    else:
        return wound_px_w, wound_px_h, area_pixels, mask_resized


# ─────────────────────────────────────────────────────────
# Calculate Accuracy (Dice Score)
# ─────────────────────────────────────────────────────────

def calculate_dice(pred_mask, true_mask_path, orig_h, orig_w):
    """Loads the true mask and calculates Dice overlap with the prediction."""
    true_mask = cv2.imread(true_mask_path, cv2.IMREAD_GRAYSCALE)
    if true_mask is None:
        return None
        
    # Resize true mask to match the original image size exactly
    if true_mask.shape[:2] != (orig_h, orig_w):
        true_mask = cv2.resize(true_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
    true_mask = (true_mask > 127).astype(np.float32)
    pred_mask = pred_mask.astype(np.float32)

    intersection = np.sum(pred_mask * true_mask)
    total_area = np.sum(pred_mask) + np.sum(true_mask)
    
    if total_area == 0:
        return 1.0  # Both masks are completely blank (perfect match on healthy skin)
        
    dice = (2.0 * intersection) / (total_area + 1e-8)
    return dice


# ─────────────────────────────────────────────────────────
# Visualise and save result
# ─────────────────────────────────────────────────────────

def visualise_result(original_bgr, mask, probability_map,
                     length, width, area, image_path, pixels_per_cm, dice_score=None):
                     
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = original_rgb.shape[:2]

    mask_full = cv2.resize(
        mask.astype(np.uint8),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST
    )

    overlay         = original_rgb.copy()
    wound_pixels    = mask_full == 1
    
    if wound_pixels.any():
        overlay[wound_pixels] = [
            int(0.4 * overlay[wound_pixels, 0].mean() + 0.6 * 220),
            int(0.4 * overlay[wound_pixels, 1].mean()),
            int(0.4 * overlay[wound_pixels, 2].mean()),
        ]
        blended = cv2.addWeighted(original_rgb, 0.6, overlay, 0.4, 0)
        contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (255, 80, 80), 2)
    else:
        blended = original_rgb.copy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    title = "Wound Analysis Result"
    if dice_score is not None:
        title += f" (Accuracy: {dice_score*100:.1f}%)"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(probability_map, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[1].set_title("Wound probability map")
    axes[1].axis("off")

    axes[2].imshow(blended)
    axes[2].set_title("Segmentation overlay")
    axes[2].axis("off")

    unit = "cm" if pixels_per_cm else "px"
    if length is not None:
        measurement_text = f"Length: {length:.2f} {unit}\nWidth:  {width:.2f} {unit}\nArea:   {area:.2f} {unit}²"
        fig.text(0.5, 0.02, measurement_text, ha="center", fontsize=12, bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    else:
        fig.text(0.5, 0.02, "No wound detected", ha="center", fontsize=12)

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    os.makedirs("predictions", exist_ok=True)
    base_name   = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join("predictions", f"{base_name}_result.png")
    
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig) # <-- CRITICAL: closes the plot memory so your RAM doesn't crash on 1000 images!

    return output_path


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict wound segmentation on a folder")
    parser.add_argument(
        "--image_folder", type=str, 
        default=r"data/test_images", 
        help="Folder containing images to test"
    )
    parser.add_argument(
        "--mask_folder", type=str, 
        default=None, 
        help="Optional: Folder containing true masks to calculate accuracy"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Confidence threshold 0-1 (default 0.5)"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  Wound Segmentation — Batch Prediction")
    print("=" * 50)

    model = load_model()

    # Find all images in the folder
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(args.image_folder) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"❌ No images found in {args.image_folder}")
        exit()

    print(f"\nFound {len(image_files)} images. Starting batch processing...")
    
    dice_scores = []

    # Loop through every image using tqdm for a nice progress bar
    for img_file in tqdm(image_files, desc="Processing Images"):
        img_path = os.path.join(args.image_folder, img_file)
        
        # 1. Preprocess
        tensor, original, orig_h, orig_w = preprocess_image(img_path)

        # 2. Predict
        mask, prob_map = predict_mask(model, tensor, threshold=args.threshold)
        prob_map_full = cv2.resize(prob_map, (orig_w, orig_h))

        # 3. Measure
        length, width, area, mask_full = measure_wound(mask, orig_h, orig_w)

        # 4. Calculate Accuracy (if masks provided)
        current_dice = None
        if args.mask_folder:
            # Assumes the mask has the exact same filename as the image
            mask_path = os.path.join(args.mask_folder, img_file)
            if os.path.exists(mask_path):
                current_dice = calculate_dice(mask_full, mask_path, orig_h, orig_w)
                if current_dice is not None:
                    dice_scores.append(current_dice)

        # 5. Save Visualisation
        visualise_result(
            original, mask_full, prob_map_full,
            length, width, area, img_path, PIXELS_PER_CM, current_dice
        )

    # Final Accuracy Report
    print("\n" + "=" * 50)
    print("  Batch Processing Complete!")
    print(f"  ✓ {len(image_files)} images saved to predictions/ folder")
    
    if args.mask_folder and dice_scores:
        avg_dice = sum(dice_scores) / len(dice_scores)
        print(f"  ✓ Overall Test Accuracy (Average Dice): {avg_dice:.4f} ({avg_dice*100:.1f}%)")
    elif args.mask_folder and not dice_scores:
        print("  ⚠ Mask folder provided, but no matching mask filenames were found.")
    else:
        print("  ℹ Provide --mask_folder to calculate accuracy/Dice scores.")
    print("=" * 50)