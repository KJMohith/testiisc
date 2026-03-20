"""Inference helpers for the DRISHTI web application."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from preprocessing.image_preprocessing import preprocess_fundus_image
from quality_check.image_quality import evaluate_image_quality

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - keeps the app usable without TensorFlow runtime.
    tf = None


CLASS_LABELS = ["Glaucoma", "Normal"]
RISK_LABELS = {
    "Normal": "Healthy",
    "Glaucoma": "Refer doctor",
}


class DrishtiClassifier:
    """Wrapper that prefers a TFLite model and falls back to a heuristic score."""

    def __init__(self, model_path: str = "tflite_model/drishti_model.tflite") -> None:
        self.model_path = Path(model_path)
        self.interpreter = None

        if tf is not None and self.model_path.exists():
            self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()[0]
            self.output_details = self.interpreter.get_output_details()[0]

    def predict(self, image_path: str) -> dict:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        quality = evaluate_image_quality(image)
        if not quality["accepted"]:
            return {
                "quality": quality,
                "label": "Rejected",
                "confidence": 0.0,
                "triage": "Retake image",
                "source": "quality_check",
            }

        if self.interpreter is not None:
            input_tensor = preprocess_fundus_image(image)
            input_tensor = np.expand_dims(input_tensor, axis=0).astype(self.input_details["dtype"])
            self.interpreter.set_tensor(self.input_details["index"], input_tensor)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details["index"])[0]
            class_index = int(np.argmax(output))
            confidence = float(output[class_index])
            label = CLASS_LABELS[class_index]
            triage = RISK_LABELS[label]
            return {
                "quality": quality,
                "label": label,
                "confidence": confidence,
                "triage": triage,
                "source": "tflite",
            }

        # Heuristic fallback so the MVP website remains runnable before training.
        processed = preprocess_fundus_image(image)
        red_mean = float(processed[:, :, 2].mean())
        green_mean = float(processed[:, :, 1].mean())
        lesion_score = max(0.0, min(1.0, (red_mean - green_mean + 0.25) * 1.5))
        label = "Glaucoma" if lesion_score >= 0.5 else "Normal"
        confidence = lesion_score if label == "Glaucoma" else 1.0 - lesion_score
        triage = "Risk" if 0.45 <= lesion_score <= 0.6 else RISK_LABELS[label]
        return {
            "quality": quality,
            "label": label,
            "confidence": float(confidence),
            "triage": triage,
            "source": "heuristic_fallback",
        }
