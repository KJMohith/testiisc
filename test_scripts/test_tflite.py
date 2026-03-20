"""Run inference on a retinal image using a TensorFlow Lite model."""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import tensorflow as tf

from preprocessing.image_preprocessing import preprocess_fundus_image

CLASS_NAMES = ["Glaucoma", "Normal"]


def run_inference(model_path: str, image_path: str) -> tuple[str, float]:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    input_tensor = preprocess_fundus_image(image)
    input_tensor = np.expand_dims(input_tensor, axis=0).astype(input_details["dtype"])

    interpreter.set_tensor(input_details["index"], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    class_index = int(np.argmax(output))
    confidence = float(output[class_index])
    return CLASS_NAMES[class_index], confidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test DRISHTI TFLite model inference.")
    parser.add_argument("--model-path", default="tflite_model/drishti_model.tflite")
    parser.add_argument("--image-path", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    label, confidence = run_inference(args.model_path, args.image_path)
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")
