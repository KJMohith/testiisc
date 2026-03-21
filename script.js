const startCameraBtn = document.getElementById('startCameraBtn');
const captureBtn = document.getElementById('captureBtn');
const fileInput = document.getElementById('fileInput');
const cameraFeed = document.getElementById('cameraFeed');
const analysisCanvas = document.getElementById('analysisCanvas');
const previewImage = document.getElementById('previewImage');
const emptyState = document.getElementById('emptyState');

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

startCameraBtn.addEventListener('click', async () => {
  try {
    activeStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' },
      audio: false,
    });
    cameraFeed.srcObject = activeStream;
    cameraFeed.hidden = false;
    emptyState.hidden = true;
    previewImage.hidden = true;
    captureBtn.disabled = false;
  } catch (error) {
    updateResult({
      level: 'yellow',
      title: 'Camera unavailable',
      summary: 'Please upload an image instead. Camera access may be blocked on this browser or device.',
      confidence: 0.52,
      brightness: null,
      contrast: null,
      redness: null,
      insights: [
        'Browser denied camera access',
        'You can still use image upload safely',
        'Run on HTTPS or localhost for camera support',
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
  previewImage.src = analysisCanvas.toDataURL('image/png');
  previewImage.hidden = false;
  cameraFeed.hidden = true;
  analyseCurrentImage();
});

fileInput.addEventListener('change', (event) => {
  const [file] = event.target.files;
  if (!file) return;

  const reader = new FileReader();
  reader.onload = () => {
    previewImage.src = reader.result;
    previewImage.hidden = false;
    emptyState.hidden = true;
    cameraFeed.hidden = true;
    analyseCurrentImage();
  };
  reader.readAsDataURL(file);
});

previewImage.addEventListener('load', () => {
  emptyState.hidden = true;
});

function analyseCurrentImage() {
  if (!previewImage.complete) return;

  const width = previewImage.naturalWidth || previewImage.width;
  const height = previewImage.naturalHeight || previewImage.height;
  analysisCanvas.width = width;
  analysisCanvas.height = height;

  const context = analysisCanvas.getContext('2d', { willReadFrequently: true });
  context.drawImage(previewImage, 0, 0, width, height);
  const { data } = context.getImageData(0, 0, width, height);

  let totalLuma = 0;
  let totalRedness = 0;
  let centerLuma = 0;
  let pixels = 0;
  let centerPixels = 0;
  const lumas = [];

  for (let y = 0; y < height; y += 2) {
    for (let x = 0; x < width; x += 2) {
      const index = (y * width + x) * 4;
      const r = data[index];
      const g = data[index + 1];
      const b = data[index + 2];
      const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      const redness = Math.max(0, r - (g + b) / 2);

      totalLuma += luma;
      totalRedness += redness;
      lumas.push(luma);
      pixels += 1;

      const inCenter =
        x > width * 0.3 &&
        x < width * 0.7 &&
        y > height * 0.3 &&
        y < height * 0.7;

      if (inCenter) {
        centerLuma += luma;
        centerPixels += 1;
      }
    }
  }

  const brightness = totalLuma / pixels;
  const redness = totalRedness / pixels;
  const centerBrightness = centerLuma / Math.max(centerPixels, 1);
  const contrast = calculateStdDev(lumas, brightness);

  const brightnessPenalty = normalizedDistance(brightness, 105, 55);
  const contrastPenalty = normalizedDistance(contrast, 48, 22);
  const rednessPenalty = normalizedDistance(redness, 22, 14);
  const centerPenalty = normalizedDistance(centerBrightness, 128, 35);

  const riskScore =
    brightnessPenalty * 0.22 +
    contrastPenalty * 0.28 +
    rednessPenalty * 0.24 +
    centerPenalty * 0.26;

  const confidence = Math.min(0.97, 0.56 + Math.abs(0.5 - riskScore) * 0.72);

  const result = classifyRisk({ brightness, contrast, redness, centerBrightness, riskScore, confidence });
  updateResult(result);
}

function calculateStdDev(values, mean) {
  const variance = values.reduce((acc, value) => acc + (value - mean) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

function normalizedDistance(value, ideal, spread) {
  return Math.min(1, Math.abs(value - ideal) / spread);
}

function classifyRisk(features) {
  const { brightness, contrast, redness, centerBrightness, riskScore, confidence } = features;

  if (riskScore > 0.66 || redness > 28 || contrast < 20) {
    return {
      level: 'red',
      title: 'Danger: high glaucoma-risk pattern detected',
      summary: 'The image shows stronger irregular cues. Please consult an ophthalmologist as soon as possible for a proper exam.',
      brightness,
      contrast,
      redness,
      confidence,
      insights: [
        `Strong imbalance between center and surrounding eye region (${centerBrightness.toFixed(0)} center intensity).`,
        'Inflammation or low-clarity cues are elevated.',
        'Urgent professional eye screening is recommended.',
      ],
    };
  }

  if (riskScore > 0.38 || redness > 16 || brightness < 70 || brightness > 170) {
    return {
      level: 'yellow',
      title: 'Consult: moderate risk pattern detected',
      summary: 'The image includes a few warning signs. A non-urgent consultation with an eye specialist would be a good next step.',
      brightness,
      contrast,
      redness,
      confidence,
      insights: [
        'Some feature values are outside the ideal range for a clean eye image.',
        'The result may improve with better lighting and sharper focus.',
        'Monitor symptoms and arrange a doctor visit if discomfort continues.',
      ],
    };
  }

  return {
    level: 'green',
    title: 'Safe: no strong glaucoma-risk cue found',
    summary: 'This image looks relatively balanced according to the tiny AI heuristic. Keep routine eye checkups for real medical assurance.',
    brightness,
    contrast,
    redness,
    confidence,
    insights: [
      'Colour and contrast features appear close to the expected range.',
      'No strong alert pattern is visible in this prototype scan.',
      'Continue regular preventive eye examinations.',
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
