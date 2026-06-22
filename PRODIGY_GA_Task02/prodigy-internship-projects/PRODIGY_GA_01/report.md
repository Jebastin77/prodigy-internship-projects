# PRODIGY_GA_01

## Task Description
Build a text generation tool using a pretrained language model. The core
task is to take a prompt typed by a user and produce a continuation of that
text using GPT-2, then wrap that capability in a simple, usable interface
instead of a bare terminal loop.

## Objective
Demonstrate prompt-based text generation with a pretrained transformer
model (GPT-2), and make it accessible through a clean local web UI built
with Django — so a user can type a prompt and immediately see the model's
generated continuation, without needing to touch the terminal or any code.

## Technologies Used
- Python
- Transformers (Hugging Face)
- PyTorch
- Django (UI layer only — no database)
- HTML / CSS for the front end

## Working
1. **Model loading** — `AutoTokenizer` and `AutoModelForCausalLM` load the
   `openai-community/gpt2` checkpoint once, when the Django server starts.
2. **User input** — the single-page UI presents a text box where the user
   types a prompt and clicks **Generate**.
3. **Tokenization** — the prompt is tokenized into input IDs with
   `tokenizer(prompt, return_tensors="pt")`.
4. **Generation** — `model.generate()` produces a continuation using
   sampling (`do_sample=True`, `temperature=0.8`) up to `max_length=100`
   tokens, padded with the EOS token.
5. **Decoding & display** — the generated token IDs are decoded back into
   text with `tokenizer.decode(...)` and rendered on the same page below
   the form, so the user sees prompt and output together.
6. The whole flow runs locally: no data is sent anywhere except to the
   in-process model running on the user's own machine.

## Results
- A working local web app where any typed prompt returns a GPT-2-generated
  continuation within the page itself.
- Because `do_sample=True`, the same prompt can produce different
  continuations on different runs — this is expected behavior of sampling-
  based generation rather than greedy decoding.
- See `screenshots/output1.png` (empty prompt screen) and
  `screenshots/output2.png` (a generated continuation) for example runs.

## Learning Outcomes
- Learned prompt engineering
- Learned text generation
- Understood tokenization
- Learned how to wrap a model-driven script into a request/response web
  flow using Django, without introducing unnecessary parts (database,
  authentication) that the task didn't need

## Author
Jebastin
