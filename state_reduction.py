import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans


# Assuming TransformerModel is implemented elsewhere and imported here
# from transformer_model import TransformerModel

class StateReductionReliability:
    def __init__(self, transformer_model, threshold=0.5, alpha=1.0, beta=1.0):


        self.transformer = transformer_model
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta

    def score_states(self, x_states):


        self.transformer.eval()
        with torch.no_grad():
            embeddings = self.transformer(x_states)  # e.g. output shape (num_states, embed_dim)
            # For scoring, we can take norm or first dimension or a dedicated output neuron
            scores = embeddings.norm(dim=1)  # Euclidean norm as importance score
        return scores, embeddings.cpu().numpy()

    def cluster_states(self, embeddings, n_clusters=10):

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_
        return cluster_labels, centroids

    def reduce_states(self, scores):
        """
        Select states with score >= threshold

        Returns:
            indices of selected states
        """
        selected_indices = (scores >= self.threshold).nonzero(as_tuple=True)[0]
        return selected_indices.cpu().numpy()

    def weighted_probability(self, P, w):

        weighted_P = P * w
        return weighted_P / np.sum(weighted_P)

    def compute_reliability_indices(self, P_k, E_k, selected_indices, total_failure_events, simulation_time):

        P_reduced = P_k[selected_indices]
        E_reduced = E_k[selected_indices]

        LOLP = np.sum(P_reduced)  # sum of failure probabilities in selected set
        EENS = np.sum(P_reduced * E_reduced)  # weighted energy not supplied
        LOLF = total_failure_events / simulation_time

        return LOLP, EENS, LOLF

    def custom_loss(self, predicted_LOLP, actual_LOLP, predicted_EENS, actual_EENS):
        """
        Custom loss combining LOLP and EENS squared errors

        Returns scalar loss
        """
        loss = self.alpha * (predicted_LOLP - actual_LOLP) ** 2 + self.beta * (predicted_EENS - actual_EENS) ** 2
        return loss


# Example usage:
if __name__ == "__main__":
    # Dummy example: replace TransformerModel with actual model
    class DummyTransformer(nn.Module):
        def __init__(self, embed_dim=64):
            super().__init__()
            self.fc = nn.Linear(embed_dim, embed_dim)

        def forward(self, x):
            # x shape: (num_states, seq_len, embed_dim)
            x = x.mean(dim=1)  # mean pooling
            return self.fc(x)  # embedding vector per state


    # Setup dummy data
    num_states = 100
    seq_len = 10
    embed_dim = 64

    transformer_model = DummyTransformer(embed_dim)
    model = StateReductionReliability(transformer_model, threshold=5.0, alpha=1.0, beta=1.0)

    # Random tensor input simulating states (num_states, seq_len, embed_dim)
    x_states = torch.randn(num_states, seq_len, embed_dim)

    # Score states and get embeddings
    scores, embeddings = model.score_states(x_states)

    # Cluster embeddings
    cluster_labels, centroids = model.cluster_states(embeddings, n_clusters=5)

    # Select reduced state set
    selected_indices = model.reduce_states(scores)

    # Dummy probabilities and energy not supplied for each state
    P_k = np.random.rand(num_states)
    P_k /= np.sum(P_k)  # normalize to sum=1
    E_k = np.random.rand(num_states) * 100  # energy not supplied (MWh)

    # Assume total failure events and simulation time
    total_failure_events = 50
    simulation_time = 8760  # hours in a year

    # Calculate reliability indices on reduced states
    LOLP, EENS, LOLF = model.compute_reliability_indices(P_k, E_k, selected_indices, total_failure_events,
                                                         simulation_time)

    print(f"LOLP: {LOLP:.4f}, EENS: {EENS:.2f} MWh, LOLF: {LOLF:.4f} events/year")

    # Compute loss example (dummy actual values)
    loss = model.custom_loss(LOLP, actual_LOLP=0.02, EENS=EENS, actual_EENS=500)
    print(f"Custom Loss: {loss:.4f}")
