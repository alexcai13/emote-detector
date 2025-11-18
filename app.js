const video = document.getElementById('webcam');
const detectionDiv = document.getElementById('detection');
const emoteDisplay = document.getElementById('emote-display');
const probabilityList = document.getElementById('probability-list');
const emoteOptions = document.getElementById('emoteOptions');
const startModal = document.getElementById('startModal');
const modalMessage = document.getElementById('modalMessage');
const modalStartBtn = document.getElementById('modalStartBtn');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');

const appState = {
  stream: null,
  session: null,
  sessionPromise: null,
  emoteMap: {},
  modelConfig: null,
  running: false,
  loopHandle: null,
  canvas: document.createElement('canvas'),
  inferenceIntervalMs: 200
};
appState.ctx = appState.canvas.getContext('2d');

function hideModal() {
  startModal.classList.remove('visible');
  startModal.setAttribute('aria-hidden', 'true');
}
function showModalMessage(text) {
  modalMessage.textContent = text;
}

async function ensureModelLoaded() {
  if (appState.session) return;
  if (appState.sessionPromise) {
    await appState.sessionPromise;
    return;
  }
  showModalMessage('Downloading model…');
  const promise = (async () => {
    const [configRes, emoteRes] = await Promise.all([
      fetch('models/web_model_config.json'),
      fetch('emote_map.json')
    ]);
    if (!configRes.ok) throw new Error('Missing models/web_model_config.json. Run export_onnx.py first.');
    if (!emoteRes.ok) throw new Error('Could not load emote_map.json');
    appState.modelConfig = await configRes.json();
    appState.emoteMap = await emoteRes.json();
    appState.canvas.width = appState.modelConfig.img_size;
    appState.canvas.height = appState.modelConfig.img_size;
    appState.session = await ort.InferenceSession.create('models/expr_resnet18.onnx', {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    });
    renderEmoteOptions();
  })();
  appState.sessionPromise = promise;
  try {
    await promise;
    showModalMessage('Ready when you are.');
  } catch (err) {
    console.error(err);
    showModalMessage('Error: ' + err.message);
    throw err;
  } finally {
    appState.sessionPromise = null;
  }
}

async function startCamera(fromModal = false) {
  try {
    await ensureModelLoaded();
    appState.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'user'
      }
    });
    video.srcObject = appState.stream;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    detectionDiv.textContent = 'Calibrating…';
    appState.running = true;
    runInferenceLoop();
    hideModal();
  } catch (err) {
    console.error('Camera error:', err);
    detectionDiv.textContent = 'Camera error';
    if (fromModal) {
      showModalMessage('Camera error: ' + err.message);
      startModal.classList.add('visible');
      startModal.setAttribute('aria-hidden', 'false');
    } else {
      alert('Camera error: ' + err.message);
    }
  }
}

function stopCamera() {
  appState.running = false;
  if (appState.loopHandle) {
    clearTimeout(appState.loopHandle);
    appState.loopHandle = null;
  }
  if (appState.stream) {
    appState.stream.getTracks().forEach(track => track.stop());
    appState.stream = null;
  }
  video.srcObject = null;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  detectionDiv.textContent = 'Camera stopped';
  emoteDisplay.innerHTML = '<span style="color: rgba(226,232,240,0.8);">Camera stopped</span>';
  probabilityList.innerHTML = '<p class="muted-text">Camera stopped.</p>';
}

function preprocessFrame() {
  const size = appState.modelConfig.img_size;
  const ctx = appState.ctx;
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(video, -size, 0, size, size);
  ctx.restore();
  const { data } = ctx.getImageData(0, 0, size, size);
  const floatData = new Float32Array(3 * size * size);
  const { mean, std } = appState.modelConfig;
  for (let i = 0; i < size * size; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    floatData[i] = (r - mean[0]) / std[0];
    floatData[size * size + i] = (g - mean[1]) / std[1];
    floatData[2 * size * size + i] = (b - mean[2]) / std[2];
  }
  return new ort.Tensor('float32', floatData, [1, 3, size, size]);
}

function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const exps = logits.map(v => Math.exp(v - maxLogit));
  const sum = exps.reduce((acc, v) => acc + v, 0);
  return exps.map(v => v / sum);
}

function formatLabel(label) {
  if (label === '67_emote') return '67';
  return label.replace(/_/g, ' ');
}

function renderEmoteOptions() {
  emoteOptions.innerHTML = '';
  const classes = appState.modelConfig?.classes || [];
  classes.forEach(label => {
    const card = document.createElement('div');
    card.className = 'emote-card';
    const img = document.createElement('img');
    const path = appState.emoteMap[label];
    if (path) {
      img.src = path;
      img.alt = label;
      card.appendChild(img);
    }
    const caption = document.createElement('span');
    caption.textContent = formatLabel(label);
    card.appendChild(caption);
    emoteOptions.appendChild(card);
  });
}

function updateProbabilityList(probabilities, topIdx) {
  probabilityList.innerHTML = '';
  appState.modelConfig.classes.forEach((label, idx) => {
    const pct = (probabilities[idx] * 100).toFixed(1);
    const item = document.createElement('div');
    item.className = 'probability-item';
    const header = document.createElement('div');
    header.className = 'probability-header';
    header.innerHTML = `<span>${formatLabel(label)}</span><span>${pct}%</span>`;
    const barBg = document.createElement('div');
    barBg.className = 'probability-bar-bg';
    const bar = document.createElement('div');
    bar.className = 'probability-bar';
    if (idx === topIdx) {
      bar.style.background = 'linear-gradient(135deg, #f97316, #db2777)';
    }
    bar.style.width = `${pct}%`;
    barBg.appendChild(bar);
    item.appendChild(header);
    item.appendChild(barBg);
    probabilityList.appendChild(item);
  });
}

async function runInferenceLoop() {
  if (!appState.running || !appState.session) return;
  if (video.readyState < 2) {
    appState.loopHandle = setTimeout(runInferenceLoop, appState.inferenceIntervalMs);
    return;
  }
  try {
    const tensor = preprocessFrame();
    const results = await appState.session.run({ input: tensor });
    const logits = Array.from(results.logits.data);
    const probabilities = softmax(logits);
    const topIdx = probabilities.indexOf(Math.max(...probabilities));
    const topLabel = appState.modelConfig.classes[topIdx] || 'unknown';
    detectionDiv.textContent = formatLabel(topLabel);
    const emotePath = appState.emoteMap[topLabel];
    if (emotePath) {
      emoteDisplay.innerHTML = `<img src="${emotePath}" alt="${topLabel}">`;
    } else {
      emoteDisplay.innerHTML = `<span style="color: rgba(226,232,240,0.8);">${formatLabel(topLabel)}</span>`;
    }
    updateProbabilityList(probabilities, topIdx);
  } catch (err) {
    console.error('Inference error:', err);
    detectionDiv.textContent = 'Inference error';
    alert('Inference error: ' + err.message);
    stopCamera();
    return;
  }
  if (appState.running) {
    appState.loopHandle = setTimeout(runInferenceLoop, appState.inferenceIntervalMs);
  }
}

modalStartBtn.addEventListener('click', () => startCamera(true));
startBtn.addEventListener('click', () => startCamera(false));
stopBtn.addEventListener('click', stopCamera);
