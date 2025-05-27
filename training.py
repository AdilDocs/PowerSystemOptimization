import torch
from torch.utils.data import DataLoader
from transformer_model import TransformerModel
from reliability_loss import ReliabilityLossWithLearnableWeights
from data_generator import create_dummy_dataset

def train_transformer(epochs=2000, batch_size=32, device='cpu'):
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
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} | α={weights[0]:.3f}, β={weights[1]:.3f}, γ={weights[2]:.3f}")

    return model, loss_fn


