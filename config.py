# config.py
# ─────────────────────────────────────────────────────────
# All settings live here. Change these to match your setup.
# ─────────────────────────────────────────────────────────

import os

# ── Paths ──────────────────────────────────────────────────
DATA_DIR        = "data"                    # root data folder
IMAGE_DIR       = os.path.join(DATA_DIR, "images")   # wound + foot images
MASK_DIR        = os.path.join(DATA_DIR, "masks")    # binary segmentation masks
MODEL_DIR       = "models"                  # saved model checkpoints
LOG_DIR         = "logs"                    # training logs

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(MODEL_DIR, "last_model.pth")

# ── Image settings ─────────────────────────────────────────
IMAGE_SIZE  = (512, 512)    # resize all images to this (height, width)
                            # 512x512 is a good balance of detail vs speed

# ── Training settings ──────────────────────────────────────
BATCH_SIZE      = 8         # how many images to process at once
                            # lower this (e.g. 4) if you run out of GPU memory
NUM_EPOCHS      = 50        # how many full passes through the dataset
LEARNING_RATE   = 1e-4      # how fast the model learns (don't change for now)
VAL_SPLIT       = 0.15      # 15% of data used for validation
TEST_SPLIT      = 0.10      # 10% of data used for final testing
RANDOM_SEED     = 42        # makes results reproducible

# ── Retraining settings ────────────────────────────────────
RETRAIN_EPOCHS  = 20        # fewer epochs needed when retraining
                            # because the model already knows a lot

# ── Model settings ─────────────────────────────────────────
NUM_CLASSES     = 1         # 1 = binary (wound vs not-wound)
                            # change to more if you add wound type labels

# ── Measurement settings ───────────────────────────────────
# This is used in predict.py to convert pixels → real cm
# You MUST place a reference object (e.g. a 1cm x 1cm sticker) in each photo
# Measure how many pixels wide it appears, then set this value
PIXELS_PER_CM   = 50        # example: if your 1cm reference = 50px wide
