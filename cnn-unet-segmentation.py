#!/usr/bin/env python3
"""
Image Segmentation of Martian Craters (U-Net, 48x48)

Train a U-Net on format B (full image + mask) with optional augmentation.
Saves model, curves, and predictions.

Authors: Renato Vivar, Miki Hagos
"""

import os
import argparse
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use("Agg")  # headless / non-interactive
import matplotlib.pyplot as plt


# -------------------------
# Utils
# -------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def balanced_accuracy(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    tp = K.sum(y_true * y_pred)
    tn = K.sum((1 - y_true) * (1 - y_pred))
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum(y_true * (1 - y_pred))
    sensitivity = tp / (tp + fn + K.epsilon())
    specificity = tn / (tn + fp + K.epsilon())
    return (sensitivity + specificity) / 2.0


def create_unet_model(input_shape=(48, 48, 1), lr=5e-4):
    inputs = layers.Input(shape=input_shape)
    s = layers.Lambda(lambda x: x / 255.0)(inputs)

    # Encoder
    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = layers.Dropout(0.2)(c3)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = layers.Dropout(0.3)(c4)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)

    # Decoder
    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = layers.Dropout(0.2)(c7)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = layers.Dropout(0.1)(c8)
    c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = layers.Dropout(0.1)(c9)
    c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=['accuracy', balanced_accuracy])
    return model


def make_augmentation_generator(rotation, zoom):
    return ImageDataGenerator(rotation_range=rotation, zoom_range=zoom)


def plot_curves(history, out_path):
    plt.figure(figsize=(7, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss')
    if 'balanced_accuracy' in history.history:
        plt.plot(history.history['balanced_accuracy'], label='Balanced Acc')
    if 'val_balanced_accuracy' in history.history:
        plt.plot(history.history['val_balanced_accuracy'], label='Val Balanced Acc')
    plt.title('Training Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_triptych(x, y_true, y_pred, out_path):
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs[0].imshow(x.squeeze(), cmap='gray'); axs[0].set_title('Input'); axs[0].axis('off')
    axs[1].imshow(y_true.squeeze(), cmap='gray'); axs[1].set_title('Ground Truth'); axs[1].axis('off')
    axs[2].imshow(y_pred.squeeze(), cmap='gray'); axs[2].set_title('Prediction'); axs[2].axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train U-Net for crater segmentation (48x48)."
    )
    # Data paths
    parser.add_argument("--xtrain_b", type=str, default="data/Xtrain2_b.npy")
    parser.add_argument("--ytrain_b", type=str, default="data/Ytrain2_b.npy")
    parser.add_argument("--xtest_b",  type=str, default="data/Xtest2_b.npy")

    # Training config
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)

    # Augmentation
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation.")
    parser.add_argument("--augment-factor", type=int, default=2,
                        help="How many extra augmented copies per original (only if --augment).")
    parser.add_argument("--rotation", type=float, default=20.0)
    parser.add_argument("--zoom", type=float, default=0.3)

    # Predictions / threshold
    parser.add_argument("--threshold", type=float, default=0.5)

    # Output
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--model-name", type=str, default="model_for_craters.keras")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    seed_everything(args.seed)

    # -------------------------
    # Load data
    # -------------------------
    x_train = np.load(args.xtrain_b)
    y_train = np.load(args.ytrain_b)
    x_test  = np.load(args.xtest_b)

    # Reshape to (N, 48, 48, 1)
    x_train = x_train.reshape(-1, 48, 48, 1)
    y_train = y_train.reshape(-1, 48, 48, 1)
    x_test  = x_test.reshape(-1, 48, 48, 1)

    # -------------------------
    # Augment (optional)
    # -------------------------
    if args.augment:
        datagen = make_augmentation_generator(args.rotation, args.zoom)
        num_aug = x_train.shape[0] * args.augment_factor
        X_aug, Y_aug = [], []
        for x, y in datagen.flow(x_train, y_train, batch_size=1):
            X_aug.append(x[0]); Y_aug.append(y[0])
            if len(X_aug) >= num_aug:
                break
        X_aug = np.array(X_aug); Y_aug = np.array(Y_aug)
        X_train_aug = np.concatenate([x_train, X_aug], axis=0)
        Y_train_aug = np.concatenate([y_train, Y_aug], axis=0)
    else:
        X_train_aug, Y_train_aug = x_train, y_train

    # -------------------------
    # Model + training
    # -------------------------
    model = create_unet_model(lr=args.lr)
    ckpt_path = os.path.join(args.outdir, args.model_name)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_loss", verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(args.outdir, "logs"))
    ]

    history = model.fit(
        X_train_aug, Y_train_aug,
        validation_split=args.val_split,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    # -------------------------
    # Curves
    # -------------------------
    plot_curves(history, os.path.join(args.outdir, "training_curves.png"))

    # -------------------------
    # Predictions
    # -------------------------
    preds_test = model.predict(x_test, verbose=1)
    preds_test_bin = (preds_test > args.threshold).astype(np.uint8)

    # Save predictions
    np.save(os.path.join(args.outdir, "preds_test.npy"), preds_test)
    np.save(os.path.join(args.outdir, "preds_test_bin.npy"), preds_test_bin)

    # Example sanity checks (val sample and test sample)
    # Take a validation sample from the end of the training set split
    val_start = int(X_train_aug.shape[0] * (1 - args.val_split))
    if args.val_split > 0 and val_start < X_train_aug.shape[0]:
        # Predict on one validation sample
        vx = X_train_aug[val_start]
        vy = Y_train_aug[val_start]
        vpred = model.predict(vx[None, ...])
        vbin = (vpred > args.threshold).astype(np.uint8)
        save_triptych(vx, vy, vbin[0], os.path.join(args.outdir, "val_triptych.png"))

    # Test triptych
    tx = x_test[0]
    tbin = preds_test_bin[0]
    save_triptych(tx, tx*0, tbin, os.path.join(args.outdir, "test_triptych.png"))  # no GT for test

    print(f"\nDone! Artifacts saved in: {args.outdir}")
    print(f"- Model: {ckpt_path}")
    print(f"- Curves: {os.path.join(args.outdir, 'training_curves.png')}")
    print(f"- Predictions: preds_test.npy & preds_test_bin.npy")
    print(f"- Example triptychs: val_triptych.png (if val split) & test_triptych.png")


if __name__ == "__main__":
    main()
