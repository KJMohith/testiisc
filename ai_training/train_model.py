"""TensorFlow/Keras training script for DRISHTI retinal screening."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score
from tensorflow import keras
from tensorflow.keras import layers

from preprocessing.dataset_loader import CLASS_NAMES, load_split


def build_model(num_classes: int) -> keras.Model:
    """Create a MobileNetV3Small-based image classifier."""
    inputs = keras.Input(shape=(224, 224, 3))
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    base_model = keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        pooling=None,
    )
    base_model.trainable = False

    x = augmentation(inputs)
    x = keras.applications.mobilenet_v3.preprocess_input(x * 255.0)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="drishti_mobilenetv3small")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(dataset_dir: str, output_model: str, epochs: int) -> None:
    """Train the classifier on preprocessed in-memory data."""
    train_split = load_split(Path(dataset_dir) / "train")
    val_split = load_split(Path(dataset_dir) / "val")

    if len(train_split.images) == 0 or len(val_split.images) == 0:
        raise ValueError("Training and validation data must both contain images.")

    model = build_model(num_classes=len(CLASS_NAMES))
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(output_model, monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(
        train_split.images,
        train_split.labels,
        validation_data=(val_split.images, val_split.labels),
        epochs=max(10, epochs),
        batch_size=16,
        callbacks=callbacks,
        verbose=1,
    )

    val_probs = model.predict(val_split.images, verbose=0)
    val_preds = np.argmax(val_probs, axis=1)
    precision = precision_score(val_split.labels, val_preds, average="macro", zero_division=0)
    recall = recall_score(val_split.labels, val_preds, average="macro", zero_division=0)
    accuracy = float((val_preds == val_split.labels).mean())

    print("Training complete")
    print(f"Best val_accuracy from history: {max(history.history['val_accuracy']):.4f}")
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Validation precision: {precision:.4f}")
    print(f"Validation recall: {recall:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DRISHTI retina classifier.")
    parser.add_argument("--dataset-dir", default="dataset", help="Dataset root containing train/ and val/ folders.")
    parser.add_argument("--output-model", default="models/drishti_model.keras", help="Path to save the best Keras model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Path(args.output_model).parent.mkdir(parents=True, exist_ok=True)
    train(args.dataset_dir, args.output_model, args.epochs)
