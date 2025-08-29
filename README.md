# Image Segmentation of Martian Craters using U-Net CNN

This project focuses on **image segmentation** of Martian surface images, specifically identifying **impact craters** in 48×48 grayscale images.  
It was developed as part of a course assignment in image analysis and deep learning.

---

## Problem Description
The goal is to perform **pixel-level classification**:
- Each pixel is labeled as:
  - `1` → crater
  - `0` → background
- Images are **48×48 pixels**.
- The dataset is **imbalanced** (more background pixels than crater pixels).
- Evaluation metric: **Balanced Accuracy**:contentReference[oaicite:1]{index=1}.

Two format provided: **Full-image segmentation (48×48 input and mask)** – used with **CNN (U-Net)**.

---

## Solution
We implemented a **U-Net Convolutional Neural Network (CNN)** with:
- **Encoder–decoder architecture** with skip connections.
- **Binary Cross-Entropy Loss**.
- **Adam optimizer** (learning rate `0.0005`).
- **Balanced Accuracy** metric (custom Keras metric).

To address class imbalance:
- Applied **data augmentation** using `ImageDataGenerator`:
  - Random rotation
  - Random zoom
- Training data was **doubled** with augmentations.


## Repository Structure



## Results
- **Training and validation loss curves** show good convergence.
- Balanced accuracy improves with augmentation.
- Resulting files are into the folder [results](results/).
- Project is explained in the [presentation attached](doc/Machine Learning - CNN 2 - UNET.pdf).  

---



## Running the Project

1. Clone the repo:

```bash
git clone https://github.com/renatovivar95/cnn-unet-image-segmentation.git
cd cnn-unet-image-segmentation
```

2. Setup 

```bash
# (optional) create venv
python3 -m venv .venv && source .venv/bin/activate

# install deps
pip install -r requirements.txt
```
In **requirements.txt** put write all of the following requirements:

- Python 3.9+
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn
- scikit-image
- tqdm
- xgboost (installed but not used in this version)

3. Run the code.

Baseline (no augmentation):

```bash
python3 segmentation_unet.py --epochs 40 --batch-size 32 --val-split 0.2
```

With augmentation (recommended):

```bash
python3 segmentation_unet.py \
  --augment --augment-factor 2 --rotation 20 --zoom 0.3 \
  --epochs 48 --batch-size 32 --val-split 0.2
```

4. What gets saved:

All artifacts go to --outdir (default: results/):
- model_for_craters.keras — best model (by val_loss)
- training_curves.png — loss & balanced accuracy curves
- preds_test.npy and preds_test_bin.npy — raw and thresholded test predictions
- val_triptych.png — Input | Ground Truth | Prediction (1 example)
- test_triptych.png — Input | Prediction (no GT)
- logs/ — TensorBoard logs

## Command-line parameters

You can change or add any of the following parameters:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--xtrain_b` | str | `Xtrain2_b.npy` | Path to training images (Format B). |
| `--ytrain_b` | str | `Ytrain2_b.npy` | Path to training masks (Format B). |
| `--xtest_b`  | str | `Xtest2_b.npy`  | Path to test images (Format B). |
| `--epochs` | int | `48` | Number of training epochs. |
| `--batch-size` | int | `32` | Mini-batch size. |
| `--val-split` | float | `0.2` | Fraction of training data used for validation. |
| `--lr` | float | `5e-4` | Adam optimizer learning rate. |
| `--seed` | int | `42` | Random seed for reproducibility (data shuffles & initialization). |
| `--augment` | flag | *off* | Enable data augmentation (rotation + zoom). |
| `--augment-factor` | int | `2` | Extra synthetic samples per original (total ≈ `1 + factor`). |
| `--rotation` | float | `20.0` | Max rotation angle (degrees) for augmentation. |
| `--zoom` | float | `0.3` | Zoom range for augmentation (e.g., 0.3 ≈ ±30%). |
| `--threshold` | float | `0.5` | Threshold to binarize predictions for saving/visuals. |
| `--outdir` | str | `results` | Output directory for artifacts. |
| `--model-name` | str | `model_for_craters.keras` | Filename for the saved model (inside `--outdir`). |

### Notes

- Use `python3 segmentation_unet.py` to run the script directly.  
- Or make it executable once (`chmod +x segmentation_unet.py`) and run with `./segmentation_unet.py`.  
- **Augmentation helps** on this small, imbalanced dataset — try:  
```bash
  python3 segmentation_unet.py --augment --augment-factor 2 --rotation 20 --zoom 0.3
```


