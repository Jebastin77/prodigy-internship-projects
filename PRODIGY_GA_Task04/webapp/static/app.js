/* Pix2Pix Lab — frontend logic */

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const dropzoneEmpty = document.getElementById('dropzone-empty');
const previewImg = document.getElementById('preview-img');

const runBtn = document.getElementById('run-btn');
const runBtnLabel = document.getElementById('run-btn-label');

const outputEmpty = document.getElementById('output-empty');
const outputCanvas = document.getElementById('output-canvas');
const outputImg = document.getElementById('output-img');
const patchOverlay = document.getElementById('patch-overlay');
const overlayToggle = document.getElementById('overlay-toggle');
const modeTag = document.getElementById('mode-tag');

const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');

const styleOptions = document.getElementById('style-options');
const styleNote = document.getElementById('style-note');

let selectedFile = null;
let selectedStyle = 'edges';
let modelLoaded = false;

/* ---------------- status check ---------------- */

async function checkStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    modelLoaded = data.model_loaded;

    if (modelLoaded) {
      statusDot.className = 'dot dot--live';
      statusText.textContent = `live model · ${data.checkpoint_name} · ${data.device}`;
      styleOptions.style.opacity = '0.4';
      styleOptions.style.pointerEvents = 'none';
      styleNote.textContent = 'A trained checkpoint is loaded — demo styles are ignored; real generator output is used.';
    } else {
      statusDot.className = 'dot dot--demo';
      statusText.textContent = 'demo mode · no checkpoint loaded';
    }
  } catch (e) {
    statusDot.className = 'dot dot--off';
    statusText.textContent = 'status unavailable';
  }
}
checkStatus();

/* ---------------- style chip selection ---------------- */

styleOptions.addEventListener('click', (e) => {
  const chip = e.target.closest('.style-chip');
  if (!chip) return;
  document.querySelectorAll('.style-chip').forEach(c => c.classList.remove('is-active'));
  chip.classList.add('is-active');
  selectedStyle = chip.dataset.style;
});

/* ---------------- upload handling ---------------- */

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});

['dragenter', 'dragover'].forEach(evt =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add('is-dragover');
  })
);

['dragleave', 'drop'].forEach(evt =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove('is-dragover');
  })
);

dropzone.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) handleFile(file);
});

function handleFile(file) {
  if (!file.type.startsWith('image/')) return;
  selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.hidden = false;
    dropzoneEmpty.hidden = true;
  };
  reader.readAsDataURL(file);

  runBtn.disabled = false;
}

/* ---------------- run translation ---------------- */

runBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  runBtn.disabled = true;
  runBtn.classList.add('is-loading');
  runBtnLabel.textContent = 'Translating';

  try {
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('style', selectedStyle);

    const res = await fetch('/api/translate', { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || 'Translation failed.');
    }
    const data = await res.json();
    renderResult(data);
  } catch (err) {
    alert(err.message || 'Something went wrong while translating the image.');
  } finally {
    runBtn.disabled = false;
    runBtn.classList.remove('is-loading');
    runBtnLabel.textContent = 'Translate image';
  }
});

function renderResult(data) {
  outputImg.src = data.output_image;
  outputEmpty.hidden = true;
  outputCanvas.hidden = false;

  modeTag.hidden = false;
  if (data.used_real_model) {
    modeTag.textContent = 'trained model';
    modeTag.className = 'mode-tag mode-tag--live';
  } else {
    modeTag.textContent = 'demo mode';
    modeTag.className = 'mode-tag mode-tag--demo';
  }

  drawPatchGrid(data.patch_scores);
}

/* ---------------- patch grid overlay (signature element) ---------------- */

function drawPatchGrid(scores) {
  patchOverlay.innerHTML = '';
  if (!scores || !scores.length) return;

  const grid = scores.length;
  const cell = 256 / grid;

  const ns = 'http://www.w3.org/2000/svg';
  scores.forEach((row, i) => {
    row.forEach((score, j) => {
      const rect = document.createElementNS(ns, 'rect');
      rect.setAttribute('x', j * cell);
      rect.setAttribute('y', i * cell);
      rect.setAttribute('width', cell);
      rect.setAttribute('height', cell);

      // score close to 1 -> high local detail/energy -> rendered as "reads real" (teal)
      // score close to 0 -> flat/uniform region -> rendered as "reads synthetic" (vermilion)
      const color = score > 0.5 ? '#3FA88A' : '#FF5A36';
      const opacity = 0.06 + Math.abs(score - 0.5) * 0.5;

      rect.setAttribute('fill', color);
      rect.setAttribute('fill-opacity', opacity.toFixed(3));
      rect.setAttribute('stroke', 'rgba(255,255,255,0.04)');
      rect.setAttribute('stroke-width', '0.5');

      patchOverlay.appendChild(rect);
    });
  });

  patchOverlay.style.display = overlayToggle.checked ? 'block' : 'none';
}

overlayToggle.addEventListener('change', () => {
  patchOverlay.style.display = overlayToggle.checked ? 'block' : 'none';
});
