Assignement 1st :



# README.md

# Run the code : 
    go to the folder transformer_architecture:

    run: 
        python generate.py

## Lightweight Transformer Language Model

This project implements a lightweight, domain-specific language model using a Transformer Decoder. It includes training, generation, and evaluation tools on a small, custom dataset (~3MB).

---

## Project Structure

```
project/
├── data/                   # Contains train.txt and val.txt
├── model.py                # Transformer decoder architecture
├── train.py                # Training loop with early stopping and validation
├── generate.py             # Prompt-based text generation script
├── eval.py                 # Evaluation: Perplexity, Distinct-n, Self-BLEU
├── samples/                # (Optional) Save generated text samples
├── best_model.pt           # Trained model checkpoint
├── README.md               # Project overview and instructions
└── requirements.txt        # List of dependencies
```

---

## Setup Instructions

1. **Clone the repository** and navigate to the directory:
   ```bash
   git clone <repo-url>
   cd project/
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset**:
   - Place a cleaned `.txt` corpus into the `data/` directory.
   - Create `train.txt` and `val.txt` files (approx 90/10 split).

---

## Training the Model

Run the training script to train your transformer model:
```bash
python train.py
```
This will:
- Train using next-token prediction
- Save the best model as `best_model.pt` using validation loss
- Use early stopping and gradient clipping

---

## Text Generation

Generate text with a prompt using sampling or greedy decoding:
```bash
python generate.py --prompt "Once upon a time" --strategy sampling --top_k 10 --temperature 1.0
```
Options:
- `--strategy`: `sampling` or `greedy`
- `--top_k`: limits sampling to top-k tokens
- `--temperature`: controls randomness

---

## Evaluation

Evaluate the trained model using:
```bash
python eval.py
```
Metrics:
- **Perplexity**: Measures prediction uncertainty
- **Distinct-n (1 & 2)**: Diversity of generated text
- **Self-BLEU**: Measures output repetitiveness

    Result:
        Perplexity: 2.45 | Cross-Entropy Loss: 0.8979

        Generating samples...

        Sample 1:
        The dragon was divided with it, and now she managed to find what they were my dear special in the magic arrows

        Sample 2:
        The dragon's daughters reached the door. Question: Why did the king's daughter say down? Answer: the morning t

        Sample 3:
        The dragon was to see how fast as they could taste, but he also bade marry up his way. She was brought to foll

        Sample 4:
        The dragon-castle, which she shook in his father's garden after the girl came home, when he heard something ro

        Sample 5:
        The dragon
        Context: And wrept about him, for he had the other passed to her, he set his house and low before i

        Sample 6:
        The dragon went into the lodge, and down on the ocean. When the prince reached it all around its from the bull

        Sample 7:
        The dragon returnh, and her mother kill him going. Question: Why was the wife from Chinjamato the Ring of Ulst

        Sample 8:
        The dragon was one years, is something to long so many lady, but he never said that nothing. Question: How did

        Sample 9:
        The dragon who present was as the wicked and drunk. When the brother the Prince ran straightway, he saw the sh

        Sample 10:
        The dragon was well tied up, they thought she gave a hand, who were they were gone, and when they were told hi

        🔍 Diversity & Fluency:
        Distinct-1: 0.038 | Distinct-2: 0.237 | Self-BLEU: 0.651

---

## Model Details

- **Architecture**: Transformer Decoder (2 layers, 4 heads)
- **Tokenization**: Character-level
- **Training**: Causal language modeling (next-token prediction)

---

## Customization

You can modify:
- `embed_dim`, `n_heads`, `n_layers` in `model.py`
- `block_size`, `batch_size`, `lr`, `epochs` in `train.py`

---

## Notes
- Designed for small corpora (1MB max)
- Ensure your input dataset is cleaned and domain-specific
- Supports sampling-based and greedy decoding for generation

---

## Example Outputs

Stored optionally in the `samples/` directory.

---

## License

MIT or specify your preferred license.


