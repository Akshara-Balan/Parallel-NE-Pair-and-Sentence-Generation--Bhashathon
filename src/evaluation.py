# src/evaluation.py
from transformers import AutoModel, AutoTokenizer
import torch

# Load LaBSE model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
model = AutoModel.from_pretrained("sentence-transformers/LaBSE")

def compute_similarity(en_sentence, target_sentence):
    """Compute sentence similarity using LaBSE."""
    inputs = tokenizer([en_sentence, target_sentence], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).pooler_output
    score = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0)
    return score.item()

if __name__ == "__main__":
    en_sentence = "John Doe went to Paris last year."
    target_sentence = "ജോൺ ഡോ കഴിഞ്ഞ വർഷം പാരീസിലേക്ക് പോയി."

    print(f"Similarity Score: {compute_similarity(en_sentence, target_sentence)}")
