# DRISHTI Web MVP

DRISHTI is a lightweight web-based retinal screening MVP that combines image preprocessing, image quality screening, TensorFlow training utilities, TensorFlow Lite conversion, and a simple browser UI for uploading fundus images and viewing triage outputs.

## What this project includes

- **Preprocessing pipeline** using OpenCV + NumPy:
  - circular retina cropping when a contour is found
  - Gaussian blur
  - CLAHE contrast enhancement
  - resize to `224x224`
  - normalization to `[0, 1]`
- **Image quality module**:
  - blur detection with Laplacian variance
  - brightness estimation from HSV
  - center alignment score
  - image accept/reject decision
- **AI training pipeline**:
  - MobileNetV3Small backbone pretrained on ImageNet
  - frozen base layers initially
  - global average pooling + dense head
  - data augmentation
  - early stopping + model checkpointing
  - validation accuracy / precision / recall reporting
- **Model compression**:
  - post-training quantized TensorFlow Lite export
- **TFLite test script** for command-line inference
- **Simple working website**:
  - fundus image upload/capture page
  - visual alignment overlay
  - traffic-light triage UI
  - browser-local scan history

## Folder structure

```text
DRISHTI/
├── ai_training/
│   └── train_model.py
├── app/
│   ├── app.py
│   ├── inference.py
│   ├── static/
│   │   ├── css/styles.css
│   │   └── js/app.js
│   └── templates/
│       └── index.html
├── models/
├── preprocessing/
│   ├── dataset_loader.py
│   └── image_preprocessing.py
├── quality_check/
│   └── image_quality.py
├── test_scripts/
│   └── test_tflite.py
├── tflite_model/
│   └── convert_to_tflite.py
├── uploads/
├── README.md
└── requirements.txt
```

## Dataset format

Place your dataset like this:

```text
dataset/
├── train/
│   ├── glaucoma/
│   └── normal/
└── val/
    ├── glaucoma/
    └── normal/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the website

```bash
python app/app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Train the model

```bash
python ai_training/train_model.py --dataset-dir dataset --output-model models/drishti_model.keras --epochs 10
```

## Convert the trained model to TensorFlow Lite

```bash
python tflite_model/convert_to_tflite.py --model-path models/drishti_model.keras --output-path tflite_model/drishti_model.tflite
```

The generated file target is `tflite_model/drishti_model.tflite`. Post-training quantization is enabled to keep the model compact.

## Test the TFLite model

```bash
python test_scripts/test_tflite.py --model-path tflite_model/drishti_model.tflite --image-path path/to/fundus.jpg
```

## Web app behavior before a model is trained

If `tflite_model/drishti_model.tflite` does not exist yet, the web app still runs using a small heuristic fallback classifier so the upload/result/history flow remains demoable. Once you place the real TFLite model in `tflite_model/`, the app automatically switches to real TFLite inference.

## Step-by-step development flow

1. Prepare dataset in the required folder structure.
2. Install dependencies from `requirements.txt`.
3. Train the Keras model.
4. Convert the trained model to TFLite.
5. Test the TFLite model with the CLI script.
6. Start the Flask web app.
7. Upload fundus images in the browser and review scan history.

## Notes

- This MVP currently uses two classes: `glaucoma` and `normal`.
- For production clinical use, you should validate with ophthalmologists, improve calibration, and add proper patient data security.
- The current web history is stored in browser `localStorage`, which is convenient for demos but not suitable for hospital deployment.
