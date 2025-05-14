import torch
import torch.nn as nn

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, block_size, embed_dim=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=n_heads, dropout=dropout)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.ln = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.block_size = block_size
        self.embed_dim = embed_dim

    def forward(self, x):
        B, T = x.size()
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)

        tok = self.token_emb(x)
        pos = self.pos_emb(pos)
        x = tok + pos

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        memory = torch.zeros((1, B, self.embed_dim), device=x.device)

        x = self.transformer(x.transpose(0, 1), memory, tgt_mask=tgt_mask)
        x = self.ln(x)
        logits = self.fc_out(x.transpose(0, 1))
        return logits
        
