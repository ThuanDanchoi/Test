from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/lama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def generate_code(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150)
    return tokenizer.decode(outputs[0])

