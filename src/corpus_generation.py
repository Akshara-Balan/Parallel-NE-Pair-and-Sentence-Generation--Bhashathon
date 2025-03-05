# src/corpus_generation.py
from transformers import pipeline

# Load a multilingual text generation model
generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct")

def generate_parallel_sentence(prompt):
    """Generate a parallel sentence with accurate NE translation."""
    response = generator(prompt, max_length=100, do_sample=True)
    return response[0]["generated_text"]

if __name__ == "__main__":
    prompt = "Translate the following sentence to Malayalam while ensuring Named Entities are correct: 'The teacher called Vishal and asked him to solve the problem on the board.'"
    print(generate_parallel_sentence(prompt))
