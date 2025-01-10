import time
import torch
from transformers import pipeline

# Initialize the model pipeline
model_id = "meta-llama/Llama-3.2-1B"
pipe = pipeline("text-generation", model=model_id, torch_dtype=torch.bfloat16, device_map="auto")

# Define test cases
test_prompts = [
    "Generate Python code for a Fibonacci sequence:",
    "What is the key to life?",
    "Explain the concept of recursion in programming:"
]

# Measure latency and output
results = []
for prompt in test_prompts:
    start_time = time.time()
    output = pipe(prompt, max_length=100)
    end_time = time.time()
    latency = end_time - start_time
    results.append({"prompt": prompt, "output": output[0]["generated_text"], "latency": latency})

# Display results
for result in results:
    print(f"Prompt: {result['prompt']}")
    print(f"Output: {result['output']}")
    print(f"Latency: {result['latency']} seconds\n")
