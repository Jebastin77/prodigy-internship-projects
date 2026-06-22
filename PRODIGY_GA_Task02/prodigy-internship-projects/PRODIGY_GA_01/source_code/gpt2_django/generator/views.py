from django.shortcuts import render

from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Model loading: identical to the original script, loaded once when the
# --- server starts so every request reuses the same tokenizer/model. ---
model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def generate_text(request):
    """
    Single-page view: GET shows the empty prompt form, POST runs the
    exact same generation logic as the original CLI script and renders
    the result back on the page.
    """
    prompt = ""
    output_text = None
    error = None

    if request.method == "POST":
        prompt = request.POST.get("prompt", "").strip()

        if not prompt:
            error = "Please enter a prompt before generating."
        else:
            # --- Generation logic: identical to the original script ---
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(
                **inputs,
                max_length=100,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # --- end of original logic ---

    return render(
        request,
        "generator/index.html",
        {
            "prompt": prompt,
            "output_text": output_text,
            "error": error,
        },
    )
