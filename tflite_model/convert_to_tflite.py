"""Convert a trained Keras model to a quantized TensorFlow Lite model."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf


def convert_to_tflite(model_path: str, output_path: str) -> None:
    """Apply post-training quantization and save a compact TFLite file."""
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(tflite_model)
    print(f"Saved TFLite model to {output} ({output.stat().st_size / 1024:.2f} KB)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DRISHTI Keras model to TFLite.")
    parser.add_argument("--model-path", default="models/drishti_model.keras")
    parser.add_argument("--output-path", default="tflite_model/drishti_model.tflite")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_to_tflite(args.model_path, args.output_path)
