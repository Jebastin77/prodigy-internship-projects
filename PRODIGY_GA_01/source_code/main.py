from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
while True:
    prompt = input("Enter prompt: ")
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=100,
        do_sample=True,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
