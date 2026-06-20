# PRODIGY_GA_01 — GPT-2 Prompt Generator (Django UI)

A local web app that wraps GPT-2 (`openai-community/gpt2`) in a clean, single-page
Django interface: type a prompt, click **Generate**, and the model's continuation
is shown back on the page. No database, no accounts, no internet calls at
generation time — everything runs on `localhost`.

## Folder structure

```
PRODIGY_GA_01/
├── README.md
├── report.md
├── requirements.txt
├── screenshots/
│   ├── output1.png        # empty prompt screen
│   └── output2.png        # generated output screen
└── source_code/
    ├── main.py             # original CLI script (unchanged), kept for reference
    └── gpt2_django/        # the Django project
        ├── manage.py
        ├── gpt2_django/    # project settings / urls / wsgi
        └── generator/      # app: view, urls, template
```

`source_code/main.py` is the original terminal script exactly as given —
the same model-loading and `generate()` call is reused inside
`generator/views.py`, just triggered by a web form (`POST`) instead of
`input()` / `while True`.

## Setup

```bash
cd PRODIGY_GA_01
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
cd source_code/gpt2_django
python manage.py runserver
```

Open **http://127.0.0.1:8000/** in your browser.

> First run only: `transformers` downloads the GPT-2 weights (~500 MB) from
> Hugging Face, so an internet connection is needed once. After that the
> model is cached locally and the app runs fully offline.

## How it works

1. The Django app loads `tokenizer` and `model` once, when the server starts
   (same two lines as the original script).
2. The page (`generator/templates/generator/index.html`) shows a text box
   for the prompt and a **Generate** button.
3. Submitting the form sends a `POST` request to the same view. The view
   runs the exact original generation code:
   ```python
   inputs = tokenizer(prompt, return_tensors="pt")
   outputs = model.generate(
       **inputs,
       max_length=100,
       do_sample=True,
       temperature=0.8,
       pad_token_id=tokenizer.eos_token_id
   )
   output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
   ```
4. The decoded text is rendered back into the same page, below the form.

## Notes

- `DATABASES` is intentionally empty in `settings.py` — this project never
  reads or writes to a database.
- The screenshots in `screenshots/` were captured from the actual rendered
  template (the typewriter-style fonts load from Google Fonts when you open
  the page with an internet connection; the example output text shown is
  illustrative — real GPT-2 output will vary each time since `do_sample=True`).

## Author

Jebastin
