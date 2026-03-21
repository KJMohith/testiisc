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

let activeStream = null;
let imageReady = false;
let currentObjectUrl = null;

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
  selectionNote.textContent = 'One camera image selected. Press Run scan for a quick result.';

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
    selectionNote.textContent = `Selected: ${file.name}. Press Run scan.`;
    URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl = null;
  };
  previewImage.src = currentObjectUrl;
});

runScanBtn.addEventListener('click', () => {
  if (!imageReady) return;

  runScanBtn.disabled = true;
  runScanBtn.textContent = 'Scanning...';
  selectionNote.textContent = 'Running quick analysis on the selected image...';

  requestAnimationFrame(() => {
    const result = analyseCurrentImage();
    updateResult(result);
    runScanBtn.disabled = false;
    runScanBtn.textContent = 'Run scan';
    selectionNote.textContent = 'Scan complete. You can choose another image anytime.';
  });
});

function analyseCurrentImage() {
  const sourceWidth = previewImage.naturalWidth || previewImage.width;
  const sourceHeight = previewImage.naturalHeight || previewImage.height;
  const maxDimension = 220;
  const scale = Math.min(1, maxDimension / Math.max(sourceWidth, sourceHeight));
  const width = Math.max(80, Math.round(sourceWidth * scale));
  const height = Math.max(80, Math.round(sourceHeight * scale));

  analysisCanvas.width = width;
  analysisCanvas.height = height;

  const context = analysisCanvas.getContext('2d', { willReadFrequently: true });
  context.clearRect(0, 0, width, height);
  context.drawImage(previewImage, 0, 0, width, height);

  const { data } = context.getImageData(0, 0, width, height);
  let totalLuma = 0;
  let totalLumaSquared = 0;
  let totalRedness = 0;
  let centerLuma = 0;
  let pixels = 0;
  let centerPixels = 0;

  for (let index = 0; index < data.length; index += 16) {
    const pixelIndex = index / 4;
    const x = pixelIndex % width;
    const y = Math.floor(pixelIndex / width);

    const r = data[index];
    const g = data[index + 1];
    const b = data[index + 2];
    const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    const redness = Math.max(0, r - (g + b) / 2);

    totalLuma += luma;
    totalLumaSquared += luma * luma;
    totalRedness += redness;
    pixels += 1;

    if (x > width * 0.3 && x < width * 0.7 && y > height * 0.3 && y < height * 0.7) {
      centerLuma += luma;
      centerPixels += 1;
    }
  }

  const brightness = totalLuma / pixels;
  const redness = totalRedness / pixels;
  const centerBrightness = centerLuma / Math.max(centerPixels, 1);
  const variance = Math.max(0, totalLumaSquared / pixels - brightness * brightness);
  const contrast = Math.sqrt(variance);

  const centerDifference = Math.abs(centerBrightness - brightness);

  const brightnessPenalty = rangePenalty(brightness, 75, 168, 40);
  const contrastPenalty = inverseRangePenalty(contrast, 18, 52);
  const rednessPenalty = upperRangePenalty(redness, 10, 26);
  const centerPenalty = upperRangePenalty(centerDifference, 12, 40);

  const riskScore =
    brightnessPenalty * 0.18 +
    contrastPenalty * 0.3 +
    rednessPenalty * 0.28 +
    centerPenalty * 0.24;

  const confidence = Math.min(0.93, 0.58 + Math.abs(0.42 - riskScore) * 0.48);
  return classifyRisk({
    brightness,
    contrast,
    redness,
    centerBrightness,
    centerDifference,
    riskScore,
    confidence,
  });
}

function rangePenalty(value, safeMin, safeMax, fade) {
  if (value >= safeMin && value <= safeMax) return 0;
  if (value < safeMin) return Math.min(1, (safeMin - value) / fade);
  return Math.min(1, (value - safeMax) / fade);
}

function inverseRangePenalty(value, dangerMin, safeMax) {
  if (value >= safeMax) return 0;
  if (value <= dangerMin) return 1;
  return 1 - (value - dangerMin) / (safeMax - dangerMin);
}

function upperRangePenalty(value, safeMax, dangerMax) {
  if (value <= safeMax) return 0;
  if (value >= dangerMax) return 1;
  return (value - safeMax) / (dangerMax - safeMax);
}

function classifyRisk(features) {
  const { brightness, contrast, redness, centerBrightness, centerDifference, riskScore, confidence } = features;

  if (riskScore > 0.72 || (redness > 30 && contrast < 24) || centerDifference > 42) {
    return {
      level: 'red',
      title: 'Danger: high-risk pattern detected',
      summary: 'The selected image shows stronger warning cues. Please consult an eye specialist as soon as possible.',
      brightness,
      contrast,
      redness,
      confidence,
      insights: [
        `Center-region difference is elevated (${centerDifference.toFixed(0)}).`,
        'The scan detected a combination of strong redness or weak contrast cues.',
        'This prototype recommends urgent medical follow-up.',
      ],
    };
  }

  if (riskScore > 0.34 || redness > 18 || contrast < 32 || centerDifference > 24) {
    return {
      level: 'yellow',
      title: 'Consult: moderate-risk pattern detected',
      summary: 'Some warning cues are present. A doctor consultation is recommended, but it is not necessarily urgent.',
      brightness,
      contrast,
      redness,
      confidence,
      insights: [
        'Some image features are outside the safer range, but not severe enough for a red alert.',
        'Better lighting or a sharper photo may improve scan quality.',
        'Consult a specialist if symptoms persist or worsen.',
      ],
    };
  }

  return {
    level: 'green',
    title: 'Safe: no strong warning cue found',
    summary: 'The selected image looks relatively balanced in this quick screening pass.',
    brightness,
    contrast,
    redness,
    confidence,
    insights: [
      'Exposure and colour balance look close to the expected range.',
      'No strong caution pattern was found in the center region.',
      'Continue routine eye checkups for proper medical care.',
    ],
  };
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
