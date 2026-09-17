import os
from config import IMAGE_DIR, MASK_DIR

print("\n=== DIAGNOSTIC CHECK ===")
print(f"Looking for Images in: {os.path.abspath(IMAGE_DIR)}")
print(f"Looking for Masks in:  {os.path.abspath(MASK_DIR)}\n")

# Check Images
if not os.path.exists(IMAGE_DIR):
    print("❌ ERROR: Image directory does not exist at this path!")
else:
    images = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]
    print(f"✅ Found {len(images)} files in Image directory.")
    if images:
        print("   First 3 image filenames exactly as Python sees them:")
        for img in images[:3]:
            print(f"     - '{img}'")

print("-" * 30)

# Check Masks
if not os.path.exists(MASK_DIR):
    print("❌ ERROR: Mask directory does not exist at this path!")
else:
    masks = [f for f in os.listdir(MASK_DIR) if os.path.isfile(os.path.join(MASK_DIR, f))]
    print(f"✅ Found {len(masks)} files in Mask directory.")
    if masks:
        print("   First 3 mask filenames exactly as Python sees them:")
        for mask in masks[:3]:
            print(f"     - '{mask}'")
            
print("\n========================")