"""
Markov Chain Text Generator — Flask App
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template_string, request, jsonify
from markov_generator import MarkovChain

app = Flask(__name__)

# ── Pre-load default dataset ───────────────────────────────────────────────
_chain = MarkovChain(order=2)
_default_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "sample_text.txt")
if os.path.exists(_default_path):
    with open(_default_path) as f:
        _chain.train(f.read())

# ── HTML (single-file, no templates folder needed) ────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Markov Text</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:      #0b0e1a;
    --surface: #131728;
    --card:    #1a1f36;
    --border:  #252c4a;
    --accent:  #6c63ff;
    --accent2: #00d4aa;
    --text:    #e8eaf6;
    --muted:   #7b82b0;
    --error:   #ff6b6b;
  }

  body {
    font-family: 'Space Grotesk', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* ── Header ── */
  header {
    border-bottom: 1px solid var(--border);
    padding: 1.25rem 2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    background: var(--surface);
  }
  .logo-icon {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
  }
  header h1 { font-size: 1.15rem; font-weight: 700; letter-spacing: -0.02em; }
  header span { font-size: 0.78rem; color: var(--muted); font-weight: 300; }
  .badge {
    margin-left: auto;
    background: linear-gradient(135deg, var(--accent)22, var(--accent2)22);
    border: 1px solid var(--accent);
    color: var(--accent);
    font-size: 0.7rem; font-weight: 600;
    padding: 0.25rem 0.65rem; border-radius: 20px;
    text-transform: uppercase; letter-spacing: 0.06em;
  }

  /* ── Layout ── */
  main {
    flex: 1;
    display: grid;
    grid-template-columns: 380px 1fr;
    gap: 0;
    max-width: 1300px;
    margin: 0 auto;
    width: 100%;
    padding: 2rem;
    gap: 1.5rem;
  }

  /* ── Cards ── */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
  }
  .card-title {
    font-size: 0.72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.1em;
    color: var(--muted); margin-bottom: 1.25rem;
    display: flex; align-items: center; gap: 0.5rem;
  }
  .card-title::before {
    content: ''; display: block;
    width: 3px; height: 14px;
    background: linear-gradient(var(--accent), var(--accent2));
    border-radius: 2px;
  }

  /* ── Controls ── */
  .controls { display: flex; flex-direction: column; gap: 1.1rem; }

  label { display: block; font-size: 0.82rem; color: var(--muted); margin-bottom: 0.4rem; font-weight: 500; }

  textarea, input[type=text] {
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    color: var(--text);
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.9rem;
    padding: 0.75rem 1rem;
    resize: vertical;
    transition: border-color 0.2s;
    outline: none;
  }
  textarea:focus, input[type=text]:focus { border-color: var(--accent); }
  textarea { min-height: 130px; }

  .slider-row { display: flex; align-items: center; gap: 0.75rem; }
  input[type=range] {
    flex: 1; accent-color: var(--accent);
    height: 4px; cursor: pointer;
  }
  .slider-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem; color: var(--accent2);
    min-width: 2.5rem; text-align: right;
  }

  select {
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    color: var(--text);
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.9rem;
    padding: 0.7rem 1rem;
    outline: none;
    cursor: pointer;
    transition: border-color 0.2s;
  }
  select:focus { border-color: var(--accent); }

  .btn {
    width: 100%; padding: 0.85rem;
    background: linear-gradient(135deg, var(--accent), #8b63ff);
    border: none; border-radius: 12px;
    color: #fff; font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem; font-weight: 600;
    cursor: pointer; transition: opacity 0.2s, transform 0.15s;
    letter-spacing: 0.01em;
  }
  .btn:hover { opacity: 0.88; transform: translateY(-1px); }
  .btn:active { transform: translateY(0); }
  .btn:disabled { opacity: 0.45; cursor: not-allowed; transform: none; }

  /* ── Output panel ── */
  .output-panel { display: flex; flex-direction: column; gap: 1.25rem; }

  .output-box {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem; line-height: 1.75;
    color: var(--text);
    min-height: 200px;
    position: relative;
    overflow-y: auto;
    transition: border-color 0.3s;
  }
  .output-box.has-text { border-color: var(--accent2)55; }
  .placeholder-msg {
    position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
    color: var(--muted); font-size: 0.9rem; text-align: center;
    pointer-events: none;
  }

  /* ── Stats row ── */
  .stats-row {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
  }
  .stat-chip {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.85rem;
    text-align: center;
  }
  .stat-chip .val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem; font-weight: 600;
    color: var(--accent2);
    display: block;
  }
  .stat-chip .lbl {
    font-size: 0.7rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.07em;
    margin-top: 0.2rem;
  }

  /* word-highlight animation */
  @keyframes pop { from { opacity:0; transform: scale(0.9); } to { opacity:1; transform: scale(1); } }
  .word-token { display: inline; }
  .word-token.new { animation: pop 0.25s ease; color: var(--accent2); transition: color 1.5s; }

  /* loading pulse */
  .pulse { animation: pulse 1s ease-in-out infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

  /* copy btn */
  .copy-btn {
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--muted);
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.78rem; padding: 0.4rem 0.8rem;
    cursor: pointer; transition: all 0.2s;
  }
  .copy-btn:hover { border-color: var(--accent2); color: var(--accent2); }

  .out-header {
    display: flex; align-items: center;
    justify-content: space-between;
    margin-bottom: 0.5rem;
  }

  @media (max-width: 800px) {
    main { grid-template-columns: 1fr; padding: 1rem; }
  }
</style>
</head>
<body>

<header>
  <div class="logo-icon">🔗</div>
  <div>
    <h1>Markov Text</h1>
    <span>Statistical language model · n-gram text generation</span>
  </div>
 
</header>

<main>
  <!-- ── Left: Controls ── -->
  <div class="card">
    <p class="card-title">Configuration</p>
    <div class="controls">

      <div>
        <label>Training Text</label>
        <textarea id="trainText" placeholder="Paste your training corpus here, or leave blank to use the default dataset…"></textarea>
      </div>

      <div>
        <label>Seed Phrase <span style="color:var(--muted);font-weight:300">(optional)</span></label>
        <input type="text" id="seedText" placeholder="e.g. The quick brown">
      </div>

      <div>
        <label>Chain Order (n-gram size)
          <span style="font-size:0.75rem;color:var(--muted)"> — higher = more coherent</span>
        </label>
        <div class="slider-row">
          <input type="range" id="orderSlider" min="1" max="4" value="2"
                 oninput="document.getElementById('orderVal').textContent=this.value">
          <span class="slider-val" id="orderVal">2</span>
        </div>
      </div>

      <div>
        <label>Output Length (words)</label>
        <div class="slider-row">
          <input type="range" id="lengthSlider" min="20" max="200" value="60" step="10"
                 oninput="document.getElementById('lengthVal').textContent=this.value">
          <span class="slider-val" id="lengthVal">60</span>
        </div>
      </div>

      <button class="btn" id="genBtn" onclick="generate()">⚡ Generate Text</button>
    </div>
  </div>

  <!-- ── Right: Output ── -->
  <div class="output-panel">
    <div class="card" style="flex:1;display:flex;flex-direction:column;">
      <div class="out-header">
        <p class="card-title" style="margin-bottom:0">Generated Output</p>
        <button class="copy-btn" onclick="copyText()">Copy</button>
      </div>
      <div class="output-box" id="outputBox">
        <span class="placeholder-msg" id="placeholder">
          Hit <strong>Generate Text</strong> to start — the model will build a Markov chain from your corpus and produce new text.
        </span>
        <span id="outputText"></span>
      </div>
    </div>

    <div class="stats-row">
      <div class="stat-chip">
        <span class="val" id="statStates">—</span>
        <div class="lbl">Model States</div>
      </div>
      <div class="stat-chip">
        <span class="val" id="statTrans">—</span>
        <div class="lbl">Transitions</div>
      </div>
      <div class="stat-chip">
        <span class="val" id="statWords">—</span>
        <div class="lbl">Words Out</div>
      </div>
    </div>
  </div>
</main>

<script>
async function generate() {
  const btn = document.getElementById('genBtn');
  const box = document.getElementById('outputBox');
  const out = document.getElementById('outputText');
  const ph  = document.getElementById('placeholder');

  btn.disabled = true;
  btn.textContent = 'Generating…';
  out.innerHTML = '<span class="pulse" style="color:var(--muted)">Building chain…</span>';
  ph.style.display = 'none';
  box.classList.remove('has-text');

  const payload = {
    text:   document.getElementById('trainText').value.trim(),
    seed:   document.getElementById('seedText').value.trim(),
    order:  parseInt(document.getElementById('orderSlider').value),
    length: parseInt(document.getElementById('lengthSlider').value),
  };

  try {
    const res  = await fetch('/generate', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    const data = await res.json();

    if (data.error) {
      out.innerHTML = '<span style="color:var(--error)">⚠ ' + data.error + '</span>';
    } else {
      // animate words in
      const words = data.text.split(' ');
      out.innerHTML = '';
      words.forEach((w, i) => {
        const span = document.createElement('span');
        span.className = 'word-token new';
        span.textContent = (i ? ' ' : '') + w;
        span.style.animationDelay = Math.min(i * 25, 600) + 'ms';
        out.appendChild(span);
        setTimeout(() => span.classList.remove('new'), Math.min(i * 25, 600) + 1500);
      });
      box.classList.add('has-text');
      document.getElementById('statStates').textContent = data.stats.states;
      document.getElementById('statTrans').textContent  = data.stats.transitions;
      document.getElementById('statWords').textContent  = words.length;
    }
  } catch(e) {
    out.innerHTML = '<span style="color:var(--error)">⚠ Request failed. Is the server running?</span>';
  }

  btn.disabled = false;
  btn.textContent = '⚡ Generate Text';
}

function copyText() {
  const t = document.getElementById('outputText').textContent.trim();
  if (t) navigator.clipboard.writeText(t).then(() => {
    const b = document.querySelector('.copy-btn');
    b.textContent = 'Copied!';
    setTimeout(() => b.textContent = 'Copy', 1500);
  });
}
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    text   = data.get("text", "").strip()
    seed   = data.get("seed", "")
    order  = max(1, min(4, int(data.get("order", 2))))
    length = max(10, min(300, int(data.get("length", 60))))

    chain = MarkovChain(order=order)

    if text:
        chain.train(text)
    else:
        chain.train(open(_default_path).read() if os.path.exists(_default_path) else "")

    if not chain.model:
        return jsonify({"error": "Not enough training text. Add more content."})

    return jsonify({
        "text":  chain.generate(length=length, seed=seed or None),
        "stats": chain.stats(),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
