# 1_setup.py
# ─────────────────────────────────────────────────────────
# Run this FIRST to check your environment is ready.
# It checks your GPU, installs missing packages, and creates
# the folder structure you need.
# ─────────────────────────────────────────────────────────

import subprocess
import sys
import os

def install(package):
    """Install a Python package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

print("=" * 50)
print("  Wound Segmentation — Environment Setup")
print("=" * 50)

# ── Step 1: Install required packages ──────────────────────
print("\n[1/4] Installing required packages...")
packages = [
    "torch",           # deep learning framework
    "torchvision",     # image utilities for PyTorch
    "segmentation-models-pytorch",  # pre-built U-Net architectures
    "albumentations",  # image augmentation (makes training more robust)
    "opencv-python",   # image loading and processing
    "numpy",           # numerical operations
    "matplotlib",      # plotting results
    "scikit-learn",    # dataset splitting utilities
    "tqdm",            # progress bars during training
    "Pillow",          # image file handling
]
for pkg in packages:
    try:
        install(pkg)
        print(f"  ✓ {pkg}")
    except Exception as e:
        print(f"  ✗ Failed to install {pkg}: {e}")

# ── Step 2: Check GPU availability ─────────────────────────
print("\n[2/4] Checking GPU...")
try:
    import torch
    if torch.cuda.is_available():
        gpu_name  = torch.cuda.get_device_name(0)
        gpu_mem   = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✓ GPU found: {gpu_name}")
        print(f"  ✓ GPU memory: {gpu_mem:.1f} GB")
        if gpu_mem < 4:
            print("  ⚠ Less than 4GB GPU RAM detected.")
            print("    Reduce BATCH_SIZE to 4 in config.py")
    else:
        print("  ⚠ No GPU detected — training will use CPU (very slow).")
        print("    Consider using Google Colab for faster training.")
except ImportError:
    print("  ✗ PyTorch not installed correctly. Try: pip install torch")

# ── Step 3: Create folder structure ────────────────────────
print("\n[3/4] Creating project folders...")
folders = [
    "data/images",    # put your wound + foot photos here
    "data/masks",     # put your binary mask images here
    "models",         # trained models will be saved here
    "logs",           # training logs saved here
    "predictions",    # output predictions saved here
]
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"  ✓ {folder}/")

# ── Step 4: Summary ────────────────────────────────────────
print("\n[4/4] Setup complete!")
print("""
Next steps:
  1. Copy your wound images  → data/images/
  2. Copy your mask images   → data/masks/
     (masks must have the same filename as the image, e.g.
      image: data/images/wound_001.jpg
      mask:  data/masks/wound_001.png)
  3. Run: python 2_prepare_data.py
""")
