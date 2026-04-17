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
    print("Preparing data for GLOBAL SHAP analysis...")
    dataset = load_all_data()
    
    # 1. We increase the batch size to 1000 to get a global statistical view
    loader = DataLoader(dataset, batch_size=5000, shuffle=True)

    # Background data for SHAP explainer (baseline)
    background_data, _ = next(iter(loader))

    # Test data to be analyzed (1000 sequences!)
    test_data, test_labels = next(iter(loader))

    print("Loading the trained ChaosOracle model...")
    model = ChaosOracle()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "models", "chaos_oracle_weights.pth")
    
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Wrap the model
    shap_model = ShapWrapper(model)

    print("Starting HEAVY SHAP computation (This might take 1-2 minutes)...")
    
    explainer = shap.DeepExplainer(shap_model, background_data)
    shap_values = explainer.shap_values(test_data)

    print("Generating Global SHAP Bar Plot...")
    feature_names = [f"Spacing {i+1}" for i in range(50)]

    if isinstance(shap_values, list):
        shap_values_np = np.array(shap_values[0])
    else:
        shap_values_np = np.array(shap_values)

    # Enforce strict 2D shape (1000 sequences, 50 features)
    shap_values_to_plot = shap_values_np.reshape(5000, 50)

    plt.figure(figsize=(10, 8))
    
    # NEW: We use plot_type="bar" to get the absolute global average
    shap.summary_plot(shap_values_to_plot, test_data.numpy(), feature_names=feature_names, plot_type="bar", show=False)
    
    # Save the plot
    graph_dir = os.path.join(base_dir, "graph")
    os.makedirs(graph_dir, exist_ok=True)
    graph_path = os.path.join(graph_dir, "shap_global_summary.png")
    
    plt.tight_layout()
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Global SHAP bar plot successfully saved at: {graph_path}")