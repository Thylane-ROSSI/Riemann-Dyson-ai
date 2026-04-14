import os
import torch
import torch.nn as nn
import shap
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Local project imports
from ml.model import ChaosOracle
from ml.train import load_all_data

class ShapWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Revert the .squeeze() from the original model's output
        # Reshape from [batch_size] to [batch_size, 1] for SHAP compatibility
        return self.model(x).unsqueeze(1)

if __name__ == "__main__":
    print("Preparing data for SHAP analysis...")
    dataset = load_all_data()
    loader = DataLoader(dataset, batch_size=100, shuffle=True)

    # 1. Background data for SHAP explainer
    background_data, _ = next(iter(loader))

    # 2. Test data to be analyzed
    test_data, test_labels = next(iter(loader))
    test_data = test_data[:50]

    print("Loading the trained ChaosOracle model...")
    model = ChaosOracle()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "models", "chaos_oracle_weights.pth")
    
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Wrap the model to ensure compatibility with SHAP's expected output dimensions
    shap_model = ShapWrapper(model)

    print("Starting SHAP computation...")
    
    # Initialize DeepExplainer with the wrapped model and background distribution
    explainer = shap.DeepExplainer(shap_model, background_data)
    shap_values = explainer.shap_values(test_data)

    print("Generating SHAP summary plot...")
    feature_names = [f"Spacing {i+1}" for i in range(50)]

    # Handle SHAP output formatting depending on the library version
    if isinstance(shap_values, list):
        shap_values_np = np.array(shap_values[0])
    else:
        shap_values_np = np.array(shap_values)

    print(f"Raw SHAP values shape: {shap_values_np.shape}")

    # Enforce strict 2D shape (50 sequences, 50 features)
    shap_values_to_plot = shap_values_np.reshape(50, 50)

    plt.figure(figsize=(10, 8))
    
    # Generate the summary plot
    shap.summary_plot(shap_values_to_plot, test_data.numpy(), feature_names=feature_names, show=False)
    
    # Save the plot
    graph_dir = os.path.join(base_dir, "graph")
    os.makedirs(graph_dir, exist_ok=True)
    graph_path = os.path.join(graph_dir, "shap_summary.png")
    
    plt.tight_layout()
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"SHAP plot successfully saved at: {graph_path}")
