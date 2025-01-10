from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

#input_text = "Generate Python code for a factorial function:"
#inputs = tokenizer(input_text, return_tensors="pt")
#outputs = model.generate(**inputs)
#print(tokenizer.decode(outputs[0]))

#from datasets import load_dataset
#dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
#for text in dataset:
    #inputs = tokenizer(text["text"], return_tensors="pt")
    #outputs = model.generate(**inputs)


from fastapi import FastAPI

app = FastAPI()

@app.post("/predict")
def predict(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    return {"output": tokenizer.decode(outputs[0])}
