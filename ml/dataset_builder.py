import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from physics.gue_generator import get_massive_gue_spacings
from mathematics.riemann_data import get_riemann_spacings


def create_sliding_windows(data_array, window_size=50):
    """Cuts a 1D array into overlapping sequences of size window_size."""
    sequences = [data_array[i : i + window_size] for i in range(len(data_array) - window_size + 1)]
    return np.array(sequences)


class ChaosDataset(Dataset):
    """PyTorch Dataset for GUE (0) vs Riemann (1) classification."""
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ==========================================
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    window_size = 50

    # 1. GUE Data (Label 0) with caching
    gue_cache_path = os.path.join(data_dir, "gue_spacings_cache.npy")
    
    if os.path.exists(gue_cache_path):
        print("Loading GUE spacings from cache")
        gue_spacings = np.load(gue_cache_path)
    else:
        print("Generating GUE matrices")
        gue_spacings = get_massive_gue_spacings(n_size=1000, nb_matrices=100)
        np.save(gue_cache_path, gue_spacings)

    # 2. Riemann Data (Label 1)
    riemann_data_path = os.path.join(data_dir, "zeros_100k.txt")
    print("Loading Riemann zeros")
    riemann_spacings = get_riemann_spacings(riemann_data_path)

    # 3. Build sliding windows and labels
    print("Building sequences")
    gue_seqs = create_sliding_windows(gue_spacings, window_size)
    riemann_seqs = create_sliding_windows(riemann_spacings, window_size)

    gue_labels = np.zeros(len(gue_seqs))      
    riemann_labels = np.ones(len(riemann_seqs)) 

    all_sequences = np.concatenate([gue_seqs, riemann_seqs], axis=0)
    all_labels = np.concatenate([gue_labels, riemann_labels], axis=0)
    
    # 4. PyTorch DataLoader test
    my_dataset = ChaosDataset(all_sequences, all_labels)
    my_dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True)
    
    print(f"Dataset ready. Total sequences: {len(my_dataset)} (GUE: {len(gue_seqs)}, Riemann: {len(riemann_seqs)})")
    
    batch_sequences, batch_labels = next(iter(my_dataloader))
    print(f"Test batch shape: {batch_sequences.shape}")