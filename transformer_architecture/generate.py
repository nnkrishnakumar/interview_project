import torch
import torch.nn.functional as F
import gradio as gr
from model import TransformerDecoder

# Load training data to create the vocabulary
with open("data/train.txt", "r") as f:
    text = f.read()
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Function to generate text based on the prompt
def generate(model, prompt, max_new_tokens=1000, temperature=1.0, top_k=10, top_p=0.8, strategy="sampling"):
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

# Gradio interface function
def generate_gradio(prompt, strategy, top_k, temperature, max_new_tokens=1000):
    model = TransformerDecoder(vocab_size, block_size=128, embed_dim=128)
    model.load_state_dict(torch.load("best_model.pt", map_location=torch.device('cpu')))
    model.eval()

    # Generate the text
    output = generate(model, prompt, max_new_tokens=max_new_tokens, strategy=strategy, top_k=top_k, temperature=temperature)
    return output

# Gradio interface setup
def create_gradio_interface():
    interface = gr.Interface(
        fn=generate_gradio,
        inputs=[
            gr.Textbox(value="Once upon a time", label="Prompt"),
            gr.Radio(["sampling", "greedy"], label="Generation Strategy", value="sampling"),
            gr.Slider(minimum=1, maximum=50, step=1, label="Top-k", value=10),
            gr.Slider(minimum=0.1, maximum=1.0, step=0.1, label="Temperature", value=1.0),
            gr.Slider(minimum=1, maximum=1000, step=1, label="Max New Tokens", value=1000)
        ],
        outputs=gr.Textbox(label="Generated Text")
    )
    return interface

# Run Gradio interface
if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(share=True)  # Set share=True to get a public URL


# import torch
# import torch.nn.functional as F
# import gradio as gr
# from model import TransformerDecoder

# # Load training data to create the vocabulary
# with open("data/train.txt", "r") as f:
#     text = f.read()
# chars = sorted(list(set(text)))
# stoi = {ch: i for i, ch in enumerate(chars)}
# itos = {i: ch for ch, i in stoi.items()}
# vocab_size = len(chars)
# encode = lambda s: [stoi[c] for c in s]
# decode = lambda l: ''.join([itos[i] for i in l])

# # Function to generate text based on the prompt
# def generate(model, prompt, max_new_tokens=1000, temperature=1.0, top_k=10, top_p=0.8, strategy="sampling"):
#     model.eval()
#     device = next(model.parameters()).device
#     context = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

#     for _ in range(max_new_tokens):
#         if context.size(1) > model.block_size:
#             context = context[:, -model.block_size:]
#         logits = model(context)
#         logits = logits[:, -1, :] / temperature
#         probs = F.softmax(logits, dim=-1)

#         if strategy == "greedy":
#             next_token = torch.argmax(probs, dim=-1)
#         else:
#             if top_k:
#                 topk_vals, topk_idx = torch.topk(probs, top_k)
#                 probs = torch.zeros_like(probs).scatter(1, topk_idx, topk_vals)
#                 probs = probs / probs.sum()
#             next_token = torch.multinomial(probs, 1)

#         context = torch.cat([context, next_token], dim=1)
#     return decode(context[0].tolist())

# # Gradio interface function
# def generate_gradio(prompt, strategy, top_k, temperature, max_new_tokens=1000):
#     model = TransformerDecoder(vocab_size, block_size=128, embed_dim=128)
#     model.load_state_dict(torch.load("best_model.pt"))
#     model.eval()

#     # Generate the text
#     output = generate(model, prompt, max_new_tokens=max_new_tokens, strategy=strategy, top_k=top_k, temperature=temperature)
#     return output

# # Gradio interface setup
# def create_gradio_interface():
#     interface = gr.Interface(
#         fn=generate_gradio,  # The function to be called
#         inputs=[
#             gr.Textbox(value="Once upon a time", label="Prompt"),  # User input prompt with 'value' instead of 'default'
#             gr.Radio(["sampling", "greedy"], label="Generation Strategy", value="sampling"),  # Sampling or greedy
#             # gr.Slider(minimum=10, maximum=10, label="Top-k", value=10),  # Top-k slider with 'value'
#             gr.Slider(minimum=0.1, maximum=1, step=1.0, label="Temperature", value=1.0),  # Temperature slider with 'value'
#             gr.Slider(minimum=1, maximum=1000, label="Max New Tokens", value=1000)  # Maximum new tokens with 'value'
#         ],
#         outputs=gr.Textbox(label="Generated Text")  # Display the generated output
#     )
#     return interface

# # Run Gradio interface
# if __name__ == "__main__":
#     interface = create_gradio_interface()
#     interface.launch()  # This will open a browser-based interface

# # File: generate.py
# import torch
# import torch.nn.functional as F
# from model import TransformerDecoder

# with open("data/train.txt", "r") as f: text = f.read()
# chars = sorted(list(set(text)))
# stoi = {ch: i for i, ch in enumerate(chars)}
# itos = {i: ch for ch, i in stoi.items()}
# vocab_size = len(chars)
# encode = lambda s: [stoi[c] for c in s]
# decode = lambda l: ''.join([itos[i] for i in l])

# def generate(model, prompt, max_new_tokens=1000, temperature=1.0, top_k=10, top_p=0.8, strategy="sampling"):
#     model.eval()
#     device = next(model.parameters()).device
#     context = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

#     for _ in range(max_new_tokens):
#         if context.size(1) > model.block_size:
#             context = context[:, -model.block_size:]
#         logits = model(context)
#         logits = logits[:, -1, :] / temperature
#         probs = F.softmax(logits, dim=-1)

#         if strategy == "greedy":
#             next_token = torch.argmax(probs, dim=-1)
#         else:
#             if top_k:
#                 topk_vals, topk_idx = torch.topk(probs, top_k)
#                 probs = torch.zeros_like(probs).scatter(1, topk_idx, topk_vals)
#                 probs = probs / probs.sum()
#             next_token = torch.multinomial(probs, 1)

#         context = torch.cat([context, next_token], dim=1)
#     return decode(context[0].tolist())

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--prompt", type=str, default="Once upon a time")
#     parser.add_argument("--strategy", type=str, default="sampling")
#     parser.add_argument("--top_k", type=int, default=10)
#     parser.add_argument("--top_p", type=float, default=None)
#     parser.add_argument("--temperature", type=float, default=1.0)
#     args = parser.parse_args()

#     model = TransformerDecoder(vocab_size, block_size=128, embed_dim=128)
#     model.load_state_dict(torch.load("best_model.pt"))
#     model.eval()

#     output = generate(model, args.prompt, top_k=args.top_k, top_p=args.top_p, temperature=args.temperature, strategy=args.strategy)
#     print("\nGenerated:\n", output)