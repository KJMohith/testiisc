"""Flask entry point for the DRISHTI web MVP."""

from __future__ import annotations

import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from app.inference import DrishtiClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
classifier = DrishtiClassifier(model_path=str(BASE_DIR / "tflite_model" / "drishti_model.tflite"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/analyze")
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded."}), 400

    file = request.files["image"]
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported image format."}), 400

    safe_name = secure_filename(file.filename or f"capture{suffix}")
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    save_path = UPLOAD_DIR / unique_name
    file.save(save_path)

    prediction = classifier.predict(str(save_path))
    prediction["filename"] = unique_name
    return jsonify(prediction)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
