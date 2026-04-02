import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from ml.dataset_builder import ChaosDataset, create_sliding_windows
from mathematics.riemann_data import get_riemann_spacings
from ml.model import ChaosOracle


def load_all_data(window_size=50):
    """Loads GUE and Riemann data, builds windows and returns the full Dataset."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")
    
    # 1. Load GUE (from cache)
    gue_cache_path = os.path.join(data_dir, "gue_spacings_cache.npy")
    gue_spacings = np.load(gue_cache_path)
    
    # 2. Load Riemann
    riemann_data_path = os.path.join(data_dir, "zeros_100k.txt")
    riemann_spacings = get_riemann_spacings(riemann_data_path)
    
    # 3. Create sequences
    gue_seqs = create_sliding_windows(gue_spacings, window_size)
    riemann_seqs = create_sliding_windows(riemann_spacings, window_size)
    
    # 4. Create labels (0 for GUE, 1 for Riemann)
    gue_labels = np.zeros(len(gue_seqs))      
    riemann_labels = np.ones(len(riemann_seqs)) 
    
    all_sequences = np.concatenate([gue_seqs, riemann_seqs], axis=0)
    all_labels = np.concatenate([gue_labels, riemann_labels], axis=0)
    
    return ChaosDataset(all_sequences, all_labels)


if __name__ == "__main__":
    print("Preparing Data")
    full_dataset = load_all_data()
    
    # 80% Train, 20% Test split
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Training on {train_size} sequences, testing on {test_size}.")

    # Initialize Model, Loss, and Optimizer
    model = ChaosOracle()
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    num_epochs = 5
    print("\nStarting Training...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for sequences, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted_classes = (outputs > 0.5).long()
            correct_predictions += (predicted_classes == labels).sum().item()
            total_predictions += labels.size(0)
            
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = (correct_predictions / total_predictions) * 100
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.2f}%")
        
    # Evaluation on Test Data
    print("\nEvaluating on Test Data")
    model.eval() 
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for sequences, labels in test_loader:
            outputs = model(sequences)
            predicted_classes = (outputs > 0.5).long()
            test_correct += (predicted_classes == labels).sum().item()
            test_total += labels.size(0)
            
    test_accuracy = (test_correct / test_total) * 100
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")

    # Save the trained model weights
    print("\nSaving Model Weights")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_dir, "chaos_oracle_weights.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model successfully saved at: {save_path}")