# PRODIGY_GA_03 — Markov Chain Text Generator

A statistical text generation model using **Markov chains**, built with Python and a sleek web UI.

## What It Does
Given a training corpus, the model builds an n-gram transition table and generates new text by probabilistically following the learned patterns.

## Setup

```bash
pip install -r requirements.txt
cd source_code
python app.py
```

Then open **http://localhost:5000** in your browser.

## Project Structure

```
PRODIGY_GA_03/
├── dataset/sample_text.txt   # Default training corpus
├── source_code/
│   ├── markov_generator.py   # Core Markov chain model
│   └── app.py                # Flask web app + UI
├── requirements.txt
└── README.md
```

## Features
- Adjustable **chain order** (1–4) controls coherence vs. randomness
- Optional **seed phrase** to guide generation
- Configurable **output length** (20–200 words)
- Live model stats: states, transitions, word count
- Paste custom text or use the bundled dataset
