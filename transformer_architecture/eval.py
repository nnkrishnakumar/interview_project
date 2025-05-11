# File: eval.py
import torch
import torch.nn.functional as F
from model import TransformerDecoder
import math
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

nltk.download('punkt')

# === Load training vocabulary ===
with open("data/train.txt", "r") as f:
    train_text = f.read()

chars = sorted(list(set(train_text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)

encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l if i in itos])

# === Evaluation functions ===
def load_data():
    with open("data/val.txt", "r") as f:
        val_text = f.read()
    return val_text

def evaluate_perplexity(model, val_text, block_size=128):
    model.eval()
    data = torch.tensor(encode(val_text), dtype=torch.long)
    total_loss, total_tokens = 0, 0
    device = next(model.parameters()).device  # Get device of the model (either CPU or CUDA)
    data = data.to(device)  # Move data to the same device as model

    with torch.no_grad():
        for i in tqdm(range(0, len(data) - block_size, block_size), desc="Evaluating"):
            x = data[i:i+block_size].unsqueeze(0)
            y = data[i+1:i+block_size+1].unsqueeze(0)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += y.numel()
    
    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)

def distinct_n(texts, n):
    total, unique = 0, set()
    for text in texts:
        tokens = list(text)
        ngrams = zip(*[tokens[i:] for i in range(n)])
        ngram_list = list(ngrams)
        total += len(ngram_list)
        unique.update(ngram_list)
    return len(unique) / total if total else 0

def self_bleu(texts):
    scores = []
    for i in range(len(texts)):
        references = [list(t) for j, t in enumerate(texts) if j != i]
        candidate = list(texts[i])
        score = sentence_bleu(references, candidate, smoothing_function=SmoothingFunction().method1)
        scores.append(score)
    return sum(scores) / len(scores)

# === Text generation using same vocabulary ===
def generate(model, prompt, max_new_tokens=100, temperature=1.0, top_k=None, strategy="sampling"):
    model.eval()
    device = next(model.parameters()).device
    context = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        if context.size(1) > model.block_size:
            context = context[:, -model.block_size:]
        logits = model(context)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)

        if strategy == "greedy":
            next_token = torch.argmax(probs, dim=-1)
        else:
            if top_k:
                topk_vals, topk_idx = torch.topk(probs, top_k)
                probs = torch.zeros_like(probs).scatter(1, topk_idx, topk_vals)
                probs = probs / probs.sum()
            next_token = torch.multinomial(probs, 1)

        context = torch.cat([context, next_token], dim=1)

    return decode(context[0].tolist())

# === Main ===
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_text = load_data()
    model = TransformerDecoder(vocab_size=vocab_size, block_size=128, embed_dim=128).to(device)  # Move model to device
    model.load_state_dict(torch.load("best_model.pt", map_location=device))  # Ensure model is loaded onto correct device
    model.eval()

    val_loss, ppl = evaluate_perplexity(model, val_text)
    print(f"\n📊 Perplexity: {ppl:.2f} | Cross-Entropy Loss: {val_loss:.4f}")

    print("\n📝 Generating samples...")
    samples = [generate(model, "The dragon", max_new_tokens=100) for _ in range(10)]
    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i}:\n{sample}")

    d1 = distinct_n(samples, 1)
    d2 = distinct_n(samples, 2)
    sb = self_bleu(samples)

    print(f"\n🔍 Diversity & Fluency:")
    print(f"Distinct-1: {d1:.3f} | Distinct-2: {d2:.3f} | Self-BLEU: {sb:.3f}")