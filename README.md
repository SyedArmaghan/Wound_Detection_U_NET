# Wound Segmentation Training Pipeline

## What this does
Trains a U-Net model specifically on wound images to:
1. Segment wounds from healthy skin
2. Measure wound length and area
3. Retrain automatically when new labeled images are added

## Files in this project
```
wound_segmentation_pipeline/
├── README.md               ← you are here
├── 1_setup.py              ← install dependencies + check GPU
├── 2_prepare_data.py       ← load, clean, and split your dataset
├── 3_train.py              ← train the U-Net model
├── 4_retrain.py            ← retrain with new data (continuous learning)
├── 5_predict.py            ← run prediction on a new image
└── config.py               ← all settings in one place
```

## How to run (in order)
```bash
python 1_setup.py           # check everything is installed
python 2_prepare_data.py    # prepare your dataset
python 3_train.py           # train from scratch
python 5_predict.py         # test on a new image
# Later, when you have new labeled images:
python 4_retrain.py         # retrain to improve
```

## Dataset folder structure expected
```
data/
├── images/                 # your wound + healthy foot images (.jpg)
└── masks/                  # binary mask for each image (same filename, .png)
                            # white = wound, black = healthy skin
```
