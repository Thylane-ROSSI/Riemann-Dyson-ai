import torch
import torch.nn as nn

class ChaosOracle(nn.Module):
    """
    Multi-Layer Perceptron (MLP) to classify sequences of level spacings.
    Input: Sequence of 50 spacings.
    Output: Probability of being Riemann (Label 1) vs GUE (Label 0).
    """
    def __init__(self, input_size=50):
        super(ChaosOracle, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.network(x)
        return out.squeeze()


if __name__ == "__main__":
    model = ChaosOracle(input_size=50)
    print("Model initialized")
    
    # Simulate a batch of 32 sequences
    dummy_input = torch.randn(32, 50)
    predictions = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape} -> Output shape: {predictions.shape}")