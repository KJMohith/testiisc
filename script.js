import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

const startCameraBtn = document.getElementById('startCameraBtn');
const captureBtn = document.getElementById('captureBtn');
const runScanBtn = document.getElementById('runScanBtn');
const fileInput = document.getElementById('fileInput');
const cameraFeed = document.getElementById('cameraFeed');
const analysisCanvas = document.getElementById('analysisCanvas');
const previewImage = document.getElementById('previewImage');
const emptyState = document.getElementById('emptyState');
const selectionNote = document.getElementById('selectionNote');

const statusCard = document.getElementById('statusCard');
const statusPill = document.getElementById('statusPill');
const statusTitle = document.getElementById('statusTitle');
const statusSummary = document.getElementById('statusSummary');
const insightList = document.getElementById('insightList');

const brightnessMetric = document.getElementById('brightnessMetric');
const contrastMetric = document.getElementById('contrastMetric');
const rednessMetric = document.getElementById('rednessMetric');
const confidenceMetric = document.getElementById('confidenceMetric');

const MODEL_ID = 'Xenova/clip-vit-base-patch32';
const MODEL_LABELS = [
  'a healthy eye',
  'a normal iris close-up',
  'a clear white sclera',
  'an irritated red eye',
  'an inflamed eye',
  'a cloudy eye',
  'an abnormal eye close-up',
  'a blurry eye photo',
];

const SAFE_LABELS = new Set(['a healthy eye', 'a normal iris close-up', 'a clear white sclera']);
const CONSULT_LABELS = new Set(['an irritated red eye', 'a blurry eye photo']);
const DANGER_LABELS = new Set(['an inflamed eye', 'a cloudy eye', 'an abnormal eye close-up']);

env.allowLocalModels = false;

let activeStream = null;
let imageReady = false;
let currentObjectUrl = null;
let classifierPromise = null;

selectionNote.textContent = 'Waiting for image selection. The open model will download on first scan.';

startCameraBtn.addEventListener('click', async () => {
  try {
    if (activeStream) {
      stopCamera();
      return;
    }

    activeStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' },
      audio: false,
    });

    cameraFeed.srcObject = activeStream;
    cameraFeed.hidden = false;
    previewImage.hidden = true;
    emptyState.hidden = true;
    captureBtn.disabled = false;
    startCameraBtn.textContent = 'Close camera';
    selectionNote.textContent = 'Camera is live. Take one photo when the eye is clear.';
  } catch (error) {
    updateResult({
      level: 'yellow',
      title: 'Camera unavailable',
      summary: 'Camera access is blocked here. Please upload a single eye image instead.',
      confidence: 0.52,
      brightness: null,
      contrast: null,
      redness: null,
      insights: [
        'Camera permission was not granted',
        'Upload works without camera support',
        'Use HTTPS or localhost if camera is required',
      ],
    });
  }
});

captureBtn.addEventListener('click', () => {
  if (!cameraFeed.videoWidth || !cameraFeed.videoHeight) return;

  analysisCanvas.width = cameraFeed.videoWidth;
  analysisCanvas.height = cameraFeed.videoHeight;
  const context = analysisCanvas.getContext('2d');
  context.drawImage(cameraFeed, 0, 0, analysisCanvas.width, analysisCanvas.height);

  previewImage.src = analysisCanvas.toDataURL('image/jpeg', 0.85);
  previewImage.hidden = false;
  cameraFeed.hidden = true;
  imageReady = true;
  runScanBtn.disabled = false;
  selectionNote.textContent = 'One camera image selected. Press Run scan to load the open model and analyse it.';

  stopCamera();
});

fileInput.addEventListener('change', (event) => {
  const [file] = event.target.files;
  if (!file) return;

  if (currentObjectUrl) {
    URL.revokeObjectURL(currentObjectUrl);
  }

  currentObjectUrl = URL.createObjectURL(file);
  previewImage.onload = () => {
    imageReady = true;
    runScanBtn.disabled = false;
    emptyState.hidden = true;
    previewImage.hidden = false;
    cameraFeed.hidden = true;
    selectionNote.textContent = `Selected: ${file.name}. Press Run scan to analyse with the open model.`;
    URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl = null;
  };
  previewImage.src = currentObjectUrl;
});

runScanBtn.addEventListener('click', async () => {
  if (!imageReady) return;

  runScanBtn.disabled = true;
  runScanBtn.textContent = 'Scanning...';
  selectionNote.textContent = 'Loading model and analysing the selected image...';

  try {
    const classifier = await getClassifier();
    const result = await analyseCurrentImage(classifier);
    updateResult(result);
    selectionNote.textContent = 'Scan complete. You can choose another image anytime.';
  } catch (error) {
    console.error(error);
    updateResult({
      level: 'yellow',
      title: 'Analysis unavailable',
      summary: 'The open model could not run in this browser session. Check the network connection and try again.',
      confidence: 0.4,
      brightness: null,
      contrast: null,
      redness: null,
      insights: [
        'The model is downloaded from a public CDN on first use',
        'A blocked network request will prevent inference',
        'Refresh and retry after the connection is available',
      ],
    });
    selectionNote.textContent = 'Analysis failed. Retry after checking network access.';
  } finally {
    runScanBtn.disabled = false;
    runScanBtn.textContent = 'Run scan';
  }
});

async function getClassifier() {
  if (!classifierPromise) {
    classifierPromise = pipeline('zero-shot-image-classification', MODEL_ID);
  }

  return classifierPromise;
}

async function analyseCurrentImage(classifier) {
  const metrics = extractImageMetrics();
  const predictions = await classifier(previewImage, MODEL_LABELS);
  return classifyRisk(predictions, metrics);
}

function extractImageMetrics() {
  const sourceWidth = previewImage.naturalWidth || previewImage.width;
  const sourceHeight = previewImage.naturalHeight || previewImage.height;
  const maxDimension = 224;
  const scale = Math.min(1, maxDimension / Math.max(sourceWidth, sourceHeight));
  const width = Math.max(96, Math.round(sourceWidth * scale));
  const height = Math.max(96, Math.round(sourceHeight * scale));

  analysisCanvas.width = width;
  analysisCanvas.height = height;

  const context = analysisCanvas.getContext('2d', { willReadFrequently: true });
  context.clearRect(0, 0, width, height);
  context.drawImage(previewImage, 0, 0, width, height);

  const { data } = context.getImageData(0, 0, width, height);
  let totalLuma = 0;
  let totalLumaSquared = 0;
  let totalRedness = 0;
  let pixels = 0;

  for (let index = 0; index < data.length; index += 16) {
    const r = data[index];
    const g = data[index + 1];
    const b = data[index + 2];
    const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    const redness = Math.max(0, r - (g + b) / 2);

    totalLuma += luma;
    totalLumaSquared += luma * luma;
    totalRedness += redness;
    pixels += 1;
  }

  const brightness = totalLuma / pixels;
  const redness = totalRedness / pixels;
  const variance = Math.max(0, totalLumaSquared / pixels - brightness * brightness);
  const contrast = Math.sqrt(variance);

  return { brightness, contrast, redness };
}

function classifyRisk(predictions, metrics) {
  const topPredictions = predictions.slice(0, 4);
  const scores = Object.fromEntries(topPredictions.map((item) => [item.label, item.score]));
  const safeScore = sumScores(scores, SAFE_LABELS);
  const consultScore = sumScores(scores, CONSULT_LABELS);
  const dangerScore = sumScores(scores, DANGER_LABELS);
  const topMatch = topPredictions[0];
  const confidence = Math.max(safeScore, consultScore, dangerScore, topMatch?.score || 0);

  if (dangerScore >= 0.5 || (dangerScore >= 0.35 && consultScore >= 0.2)) {
    return {
      level: 'red',
      title: 'Danger: strong abnormal-eye signal detected',
      summary: 'The open vision model found stronger abnormal or cloudy-eye cues in this image. Please seek medical advice promptly.',
      ...metrics,
      confidence,
      insights: [
        `Top model label: ${formatLabel(topMatch?.label)} (${formatPercent(topMatch?.score)}).`,
        `Abnormal-eye score: ${formatPercent(dangerScore)}.`,
        'This is still a demo screening result, not a medical diagnosis.',
      ],
    };
  }

  if (consultScore >= 0.45 || dangerScore >= 0.25 || (consultScore + dangerScore) >= 0.55) {
    return {
      level: 'yellow',
      title: 'Consult: review recommended',
      summary: 'The model found irritation, blur, or mild abnormal-eye cues. A clearer image or non-urgent doctor review is recommended.',
      ...metrics,
      confidence,
      insights: [
        `Top model label: ${formatLabel(topMatch?.label)} (${formatPercent(topMatch?.score)}).`,
        `Review score: ${formatPercent(consultScore + dangerScore)}.`,
        'Retake the image in good lighting if the result does not match what you expect.',
      ],
    };
  }

  return {
    level: 'green',
    title: 'Safe: mostly healthy-eye signal',
    summary: 'The open model matched this image more closely to healthy-eye descriptions than warning descriptions.',
    ...metrics,
    confidence,
    insights: [
      `Top model label: ${formatLabel(topMatch?.label)} (${formatPercent(topMatch?.score)}).`,
      `Healthy-eye score: ${formatPercent(safeScore)}.`,
      'If the photo is blurry or symptoms are present, retake the image or consult a clinician anyway.',
    ],
  };
}

function sumScores(scores, labels) {
  let total = 0;

  labels.forEach((label) => {
    total += scores[label] || 0;
  });

  return total;
}

function formatPercent(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

function formatLabel(label) {
  return label ? label.replace(/^a\s+/i, '').replace(/^an\s+/i, '') : 'no match';
}

function updateResult(result) {
  const { level, title, summary, brightness, contrast, redness, confidence, insights } = result;
  statusCard.className = `status-card ${level}`;
  statusPill.className = `status-pill ${level}`;
  statusPill.textContent = level === 'red' ? 'Danger' : level === 'yellow' ? 'Consult' : level === 'green' ? 'Safe' : 'Waiting';
  statusTitle.textContent = title;
  statusSummary.textContent = summary;

  brightnessMetric.textContent = formatMetric(brightness);
  contrastMetric.textContent = formatMetric(contrast);
  rednessMetric.textContent = formatMetric(redness);
  confidenceMetric.textContent = confidence ? `${Math.round(confidence * 100)}%` : '—';

  insightList.innerHTML = '';
  (insights || []).forEach((item) => {
    const li = document.createElement('li');
    li.textContent = item;
    insightList.appendChild(li);
  });
}

function formatMetric(value) {
  return typeof value === 'number' ? value.toFixed(1) : '—';
}

function stopCamera() {
  if (activeStream) {
    activeStream.getTracks().forEach((track) => track.stop());
    activeStream = null;
  }

  cameraFeed.srcObject = null;
  cameraFeed.hidden = true;
  captureBtn.disabled = true;
  startCameraBtn.textContent = 'Open camera';
}
