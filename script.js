import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/+esm';

env.allowLocalModels = false;

env.backends ??= {};
env.backends.onnx ??= {};
env.backends.onnx.wasm ??= {};
env.backends.onnx.wasm.proxy = false;

const MODEL_ID = 'Xenova/clip-vit-base-patch32';
const CANDIDATE_LABELS = [
  'a healthy eye',
  'a normal iris close-up',
  'a clear white sclera',
  'an irritated red eye',
  'an inflamed eye',
  'a cloudy eye',
  'an abnormal eye close-up',
  'a blurry eye photo',
];

const SAFE_LABELS = new Set([
  'a healthy eye',
  'a normal iris close-up',
  'a clear white sclera',
]);
const CONSULT_LABELS = new Set(['an irritated red eye', 'a blurry eye photo']);
const DANGER_LABELS = new Set(['an inflamed eye', 'a cloudy eye', 'an abnormal eye close-up']);

const previewShell = document.querySelector('.preview-shell');
const cameraFeed = document.getElementById('cameraFeed');
const imagePreview = document.getElementById('imagePreview');
const processingCanvas = document.getElementById('processingCanvas');
const imageInput = document.getElementById('imageInput');
const cameraToggleBtn = document.getElementById('cameraToggleBtn');
const captureBtn = document.getElementById('captureBtn');
const runBtn = document.getElementById('runBtn');
const statusNote = document.getElementById('statusNote');
const resultCard = document.getElementById('resultCard');
const resultPill = document.getElementById('resultPill');
const resultTitle = document.getElementById('resultTitle');
const resultSummary = document.getElementById('resultSummary');
const insightsList = document.getElementById('insightsList');
const brightnessMetric = document.getElementById('brightnessMetric');
const contrastMetric = document.getElementById('contrastMetric');
const rednessMetric = document.getElementById('rednessMetric');
const confidenceMetric = document.getElementById('confidenceMetric');

let cameraStream = null;
let classifierPromise = null;
let selectedImageReady = false;

const setStatus = (message) => {
  statusNote.textContent = message;
};

const setPreviewMode = (mode) => {
  previewShell.classList.toggle('has-video', mode === 'video');
  previewShell.classList.toggle('has-image', mode === 'image');
};

const updateRunAvailability = () => {
  runBtn.disabled = !selectedImageReady;
};

const resetResultCard = () => {
  resultCard.className = 'result-card state-neutral';
  resultPill.className = 'result-pill neutral';
};

const setResultState = ({ state, pill, title, summary }) => {
  resultCard.className = `result-card state-${state}`;
  resultPill.className = `result-pill ${state}`;
  resultPill.textContent = pill;
  resultTitle.textContent = title;
  resultSummary.textContent = summary;
};

const setMetric = (element, value, suffix = '') => {
  element.textContent = Number.isFinite(value) ? `${value.toFixed(1)}${suffix}` : '—';
};

const setConfidenceMetric = (value) => {
  confidenceMetric.textContent = Number.isFinite(value) ? `${value.toFixed(1)}%` : '—';
};

const setInsights = (items) => {
  insightsList.innerHTML = '';
  for (const item of items) {
    const li = document.createElement('li');
    li.textContent = item;
    insightsList.appendChild(li);
  }
};

const stopCamera = () => {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  cameraFeed.srcObject = null;
  captureBtn.disabled = true;
  cameraToggleBtn.textContent = 'Open camera';
  if (!selectedImageReady) {
    setPreviewMode('empty');
  }
};

const startCamera = async () => {
  cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
  cameraFeed.srcObject = cameraStream;
  captureBtn.disabled = false;
  cameraToggleBtn.textContent = 'Close camera';
  setPreviewMode('video');
  setStatus('Camera active. Capture one clear frame when ready.');
};

const drawImageToCanvas = (source, sourceWidth, sourceHeight) => {
  const maxDimension = 224;
  const minDimension = 96;
  const largestSide = Math.max(sourceWidth, sourceHeight);
  const scale = Math.min(1, maxDimension / largestSide);
  let targetWidth = Math.max(minDimension, Math.round(sourceWidth * scale));
  let targetHeight = Math.max(minDimension, Math.round(sourceHeight * scale));

  if (sourceWidth >= sourceHeight && targetHeight > maxDimension) {
    const readjustedScale = maxDimension / targetHeight;
    targetWidth = Math.max(minDimension, Math.round(targetWidth * readjustedScale));
    targetHeight = maxDimension;
  }

  if (sourceHeight > sourceWidth && targetWidth > maxDimension) {
    const readjustedScale = maxDimension / targetWidth;
    targetHeight = Math.max(minDimension, Math.round(targetHeight * readjustedScale));
    targetWidth = maxDimension;
  }

  processingCanvas.width = targetWidth;
  processingCanvas.height = targetHeight;
  const context = processingCanvas.getContext('2d', { willReadFrequently: true });
  context.clearRect(0, 0, targetWidth, targetHeight);
  context.drawImage(source, 0, 0, targetWidth, targetHeight);
  return context;
};

const computeImageMetrics = (context) => {
  const { data } = context.getImageData(0, 0, processingCanvas.width, processingCanvas.height);
  let brightnessSum = 0;
  let varianceAccumulator = 0;
  let rednessAccumulator = 0;
  const lumas = [];

  for (let index = 0; index < data.length; index += 4) {
    const red = data[index];
    const green = data[index + 1];
    const blue = data[index + 2];
    const luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue;
    brightnessSum += luma;
    rednessAccumulator += red - (green + blue) / 2;
    lumas.push(luma);
  }

  const pixelCount = lumas.length || 1;
  const meanBrightness = brightnessSum / pixelCount;

  for (const luma of lumas) {
    varianceAccumulator += (luma - meanBrightness) ** 2;
  }

  const contrast = Math.sqrt(varianceAccumulator / pixelCount);
  const rednessCue = rednessAccumulator / pixelCount;

  return {
    brightness: meanBrightness / 2.55,
    contrast: contrast / 2.55,
    redness: rednessCue / 2.55,
  };
};

const ensureClassifier = async () => {
  if (!classifierPromise) {
    classifierPromise = pipeline('zero-shot-image-classification', MODEL_ID, {
      progress_callback: (progress) => {
        if (progress?.status === 'progress' && Number.isFinite(progress.progress)) {
          setStatus(`Loading open model… ${Math.round(progress.progress)}%`);
        } else if (progress?.status === 'download') {
          setStatus('Downloading model files from the public CDN…');
        } else if (progress?.status === 'ready') {
          setStatus('Model ready. Analysing the selected image…');
        }
      },
    }).catch((error) => {
      classifierPromise = null;
      throw error;
    });
  }
  return classifierPromise;
};

const aggregateScores = (predictions) => {
  const totals = { safe: 0, consult: 0, danger: 0 };
  for (const prediction of predictions) {
    if (SAFE_LABELS.has(prediction.label)) totals.safe += prediction.score;
    if (CONSULT_LABELS.has(prediction.label)) totals.consult += prediction.score;
    if (DANGER_LABELS.has(prediction.label)) totals.danger += prediction.score;
  }
  return totals;
};

const mapScoresToState = ({ safe, consult, danger }) => {
  if (danger >= 0.45 || danger + consult >= 0.68) {
    return {
      state: 'red',
      pill: 'Danger',
      title: 'Danger: strong abnormal-eye signal detected',
      summary:
        'Stronger abnormal or cloudy-eye cues were found in this image, so prompt medical advice is recommended.',
      riskScore: Math.max(danger, danger + consult * 0.5),
    };
  }

  if (consult >= 0.24 || danger >= 0.2 || consult + danger >= 0.35) {
    return {
      state: 'yellow',
      pill: 'Consult',
      title: 'Consult: review recommended',
      summary:
        'Irritation, blur, or mild abnormal-eye cues were found, so a clearer image or non-urgent doctor review is recommended.',
      riskScore: Math.max(consult, danger),
    };
  }

  return {
    state: 'green',
    pill: 'Safe',
    title: 'Safe: mostly healthy-eye signal',
    summary: 'The model matched more closely to healthy-eye descriptions than warning descriptions.',
    riskScore: Math.max(0, 1 - safe),
  };
};

const loadFilePreview = (file) => {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    imagePreview.src = String(reader.result);
    imagePreview.onload = () => {
      selectedImageReady = true;
      setPreviewMode('image');
      updateRunAvailability();
      setStatus('Image ready. Press Run scan to analyse this eye image.');
    };
  };
  reader.readAsDataURL(file);
};

cameraToggleBtn.addEventListener('click', async () => {
  try {
    if (cameraStream) {
      stopCamera();
      setStatus(selectedImageReady ? 'Camera closed. Existing image remains selected.' : 'Waiting for image selection.');
      return;
    }

    await startCamera();
  } catch (error) {
    console.error(error);
    stopCamera();
    setStatus('Camera access was unavailable. You can still upload one image.');
  }
});

captureBtn.addEventListener('click', () => {
  if (!cameraStream) return;

  drawImageToCanvas(cameraFeed, cameraFeed.videoWidth || 224, cameraFeed.videoHeight || 224);
  imagePreview.src = processingCanvas.toDataURL('image/png');
  imagePreview.onload = () => {
    selectedImageReady = true;
    setPreviewMode('image');
    updateRunAvailability();
    setStatus('Camera photo captured. Press Run scan to analyse this frame.');
  };
});

imageInput.addEventListener('change', (event) => {
  const [file] = event.target.files || [];
  loadFilePreview(file);
});

runBtn.addEventListener('click', async () => {
  if (!selectedImageReady) return;

  runBtn.disabled = true;
  const originalLabel = runBtn.textContent;
  runBtn.textContent = 'Scanning...';
  setStatus('Model loading or analysing. The first scan may take longer while files download.');

  try {
    const context = drawImageToCanvas(imagePreview, imagePreview.naturalWidth || 224, imagePreview.naturalHeight || 224);
    const metrics = computeImageMetrics(context);
    setMetric(brightnessMetric, metrics.brightness);
    setMetric(contrastMetric, metrics.contrast);
    setMetric(rednessMetric, metrics.redness);

    const classifier = await ensureClassifier();
    const predictions = await classifier(processingCanvas, CANDIDATE_LABELS);
    const sortedPredictions = [...predictions].sort((left, right) => right.score - left.score);
    const topPrediction = sortedPredictions[0];
    const scores = aggregateScores(sortedPredictions);
    const outcome = mapScoresToState(scores);

    setConfidenceMetric(topPrediction.score * 100);
    setResultState(outcome);
    setInsights([
      `Top model label: ${topPrediction.label} (${(topPrediction.score * 100).toFixed(1)}%)`,
      `Aggregated risk score: ${(outcome.riskScore * 100).toFixed(1)}%`,
      'Demo only: this screening-style output is not a diagnosis.',
    ]);
    setStatus('Scan complete. Review the result and quality cues below.');
  } catch (error) {
    console.error(error);
    setConfidenceMetric(Number.NaN);
    setResultState({
      state: 'yellow',
      pill: 'Notice',
      title: 'Analysis unavailable',
      summary: 'The open model could not run. Check the network connection and try again.',
    });
    setInsights([
      'The first use downloads model files from a public CDN before inference begins.',
      'If the download is blocked or offline, browser-side analysis cannot start.',
      `Runtime detail: ${error instanceof Error ? error.message : 'Unexpected browser-side model error.'}`,
    ]);
    setStatus('Analysis unavailable. Try again after verifying the network connection.');
  } finally {
    runBtn.textContent = originalLabel;
    runBtn.disabled = false;
  }
});

window.addEventListener('beforeunload', stopCamera);
resetResultCard();
updateRunAvailability();
setInsights([
  'Open-model label matching',
  'Confidence-based scoring',
  'Basic image quality metrics',
]);
