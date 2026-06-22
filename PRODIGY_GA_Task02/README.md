# PRODIGY_GA_Task02 — AI Image Generation with Pre-trained Models

> **Prodigy InfoTech Generative AI Internship · Task 02**
> Generate images from text prompts using **Stable Diffusion v1.5** served through a **Django** web UI.

---

## 📁 Project Structure

```
PRODIGY_GA_Task02/
├── app.py                      # One-command launcher
├── manage.py                   # Django CLI
├── requirements.txt
├── README.md
├── report.md
├── db.sqlite3                  # Auto-created on first run
│
├── images/                     # Generated images saved here
│
├── prompts/
│   └── prompts.txt             # Sample optimised prompts
│
├── source_code/
│   └── image_generator.py      # Core SD pipeline wrapper
│
└── image_gen_project/          # Django project
    ├── settings.py
    ├── urls.py
    ├── wsgi.py
    ├── templates/
    │   └── generator/
    │       └── index.html      # Full-featured web UI
    └── generator/
        ├── views.py            # HTTP handlers
        └── urls.py
```

---

## ⚙ Requirements

| Tool | Minimum version |
|------|----------------|
| Python | 3.10 + |
| VRAM (GPU) | 4 GB (float16) |
| RAM (CPU) | 8 GB (float32, slow) |

---

## 🚀 Quick Start

### 1 · Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2 · Install dependencies
```bash
pip install -r requirements.txt
```

> **GPU users (recommended):** Install the CUDA build of PyTorch first:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### 3 · Run the app
```bash
python app.py
```
On first launch the **Stable Diffusion v1.5** weights (~4 GB) are downloaded
from HuggingFace and cached in `~/.cache/huggingface/`.

### 4 · Open the browser
```
http://127.0.0.1:8000
```

---

## 🎨 Using the UI

| Control | Description |
|---------|-------------|
| **Prompt** | Describe what you want. Quality boosters are appended automatically. |
| **Negative Prompt** | Concepts to exclude (blurriness, watermarks…). |
| **Inference Steps** | 10–50. More steps → higher quality, slower. |
| **Guidance Scale** | 1–20. Higher = more prompt-adherent but less creative. |
| **Width / Height** | Output resolution (512 × 512 recommended). |
| **Seed** | Leave blank for random; set a value to reproduce a result. |
| **Sample Prompts** | Click any chip to auto-fill the prompt field. |

---

## 🧠 Prompt Optimisation

`source_code/image_generator.py` automatically appends quality boosters:

```
ultra-detailed, high resolution, 8K, sharp focus,
professional photography, award-winning
```

…unless they are already present in your prompt.

---

## 📝 References

- [Stable Diffusion v1.5 on HuggingFace](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers)
- [Django Documentation](https://docs.djangoproject.com/)
