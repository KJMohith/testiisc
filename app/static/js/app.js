const imageInput = document.getElementById('imageInput');
const preview = document.getElementById('preview');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultText = document.getElementById('resultText');
const trafficLight = document.getElementById('trafficLight');
const metricsList = document.getElementById('metrics');
const historyList = document.getElementById('historyList');

let selectedFile = null;

function renderHistory() {
  const history = JSON.parse(localStorage.getItem('drishti-history') || '[]');
  historyList.innerHTML = history.length
    ? history
        .map(
          (item) => `
            <div class="history-item">
              <strong>${item.timestamp}</strong><br />
              ${item.label} · ${item.triage} · ${(item.confidence * 100).toFixed(1)}%
            </div>
          `,
        )
        .join('')
    : '<p>No scans saved yet.</p>';
}

function setTraffic(triage) {
  trafficLight.className = 'traffic';
  if (triage === 'Healthy') trafficLight.classList.add('green');
  else if (triage === 'Risk') trafficLight.classList.add('yellow');
  else if (triage === 'Refer doctor' || triage === 'Retake image') trafficLight.classList.add('red');
  else trafficLight.classList.add('neutral');
}

imageInput.addEventListener('change', (event) => {
  selectedFile = event.target.files[0] || null;
  analyzeBtn.disabled = !selectedFile;

  if (!selectedFile) {
    preview.classList.add('hidden');
    preview.src = '';
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.classList.remove('hidden');
  };
  reader.readAsDataURL(selectedFile);
});

analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  const formData = new FormData();
  formData.append('image', selectedFile);
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Analyzing...';

  try {
    const response = await fetch('/analyze', {
      method: 'POST',
      body: formData,
    });
    const data = await response.json();

    if (!response.ok) throw new Error(data.error || 'Analysis failed.');

    resultText.textContent = `${data.triage}: ${data.label} (${(data.confidence * 100).toFixed(1)}%)`;
    setTraffic(data.triage);
    metricsList.innerHTML = `
      <li><strong>Quality accepted:</strong> ${data.quality.accepted}</li>
      <li><strong>Blur score:</strong> ${data.quality.blur_score.toFixed(2)}</li>
      <li><strong>Brightness:</strong> ${data.quality.brightness_score.toFixed(2)}</li>
      <li><strong>Center alignment:</strong> ${data.quality.center_alignment_score.toFixed(2)}</li>
      <li><strong>Inference source:</strong> ${data.source}</li>
    `;

    const history = JSON.parse(localStorage.getItem('drishti-history') || '[]');
    history.unshift({
      timestamp: new Date().toLocaleString(),
      label: data.label,
      triage: data.triage,
      confidence: data.confidence,
    });
    localStorage.setItem('drishti-history', JSON.stringify(history.slice(0, 10)));
    renderHistory();
  } catch (error) {
    resultText.textContent = error.message;
    setTraffic('Retake image');
    metricsList.innerHTML = '';
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze image';
  }
});

renderHistory();
