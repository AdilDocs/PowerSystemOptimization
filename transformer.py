import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, TensorDataset


# --- Transformer components ---

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=64, heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        assert embed_dim % heads == 0, "embed_dim must be divisible by heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        Q = self.query(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=64, heads=8, ff_dim=256, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attn(x)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class TransformerModel(nn.Module):
    def __init__(self, embed_dim=64, heads=8, layers=4, ff_dim=256, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, heads, ff_dim, dropout) for _ in range(layers)])
        self.out = nn.Linear(embed_dim, 3)  # Predict LOLP, EENS, LOLF

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)  # Mean pooling across sequence length
        return self.out(x)


# --- Learnable weighted loss function ---

class ReliabilityLossWithLearnableWeights(nn.Module):
    def __init__(self):
        super().__init__()
        # Raw logits for weights, normalized by softmax during forward pass
        self.raw_weights = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))

    def forward(self, pred, target):
        weights = torch.softmax(self.raw_weights, dim=0)  # Sum to 1, positive
        alpha, beta, gamma = weights[0], weights[1], weights[2]

        loss_lolp = F.mse_loss(pred[:, 0], target[:, 0])
        loss_eens = F.mse_loss(pred[:, 1], target[:, 1])
        loss_lolf = F.mse_loss(pred[:, 2], target[:, 2])

        loss = alpha * loss_lolp + beta * loss_eens + gamma * loss_lolf
        return loss, (alpha.item(), beta.item(), gamma.item())


# --- Dummy dataset generator ---

def create_dummy_dataset(num_samples=1000, seq_len=10, embed_dim=64):
    X = torch.randn(num_samples, seq_len, embed_dim)
    y = torch.rand(num_samples, 3)  # Random LOLP, EENS, LOLF targets in [0,1]
    return TensorDataset(X, y)


# --- Training loop ---

def train_transformer(epochs=20, batch_size=32, device='cpu'):
    dataset = create_dummy_dataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TransformerModel().to(device)
    loss_fn = ReliabilityLossWithLearnableWeights().to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss, weights = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f} | α={weights[0]:.3f}, β={weights[1]:.3f}, γ={weights[2]:.3f}")



