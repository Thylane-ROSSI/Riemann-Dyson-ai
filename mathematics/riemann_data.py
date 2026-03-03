import os
import numpy as np

def load_riemann_zeros(filepath):
    """
    Loads the Riemann Zeta zeros from a text file.
    
    Parameters:
    filepath (str): Path to the text file containing the zeros.
    
    Returns:
    numpy.ndarray: Array of the zeros' heights (gamma values).
    """
    print(f"Loading data from {filepath}...")
    
    try:
        # np.loadtxt is perfect for reading columns of numbers in text files
        zeros = np.loadtxt(filepath)
        return zeros
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        print("Please make sure you downloaded it and put it in the 'data' folder.")
        return None

if __name__ == "__main__":
    # Define the path to our newly downloaded file
    data_path = os.path.join("data", "zeros_100k.txt")
    
    # Load the zeros
    riemann_zeros = load_riemann_zeros(data_path)
    
    if riemann_zeros is not None:
        num_zeros = len(riemann_zeros)
        print(f"\nSuccess! Loaded {num_zeros} Riemann zeros.")
        
        # Physics / Math check: The first zero should be around 14.1347
        print("\n--- The First 5 Zeros ---")
        for i in range(5):
            print(f"Zero {i+1}: {riemann_zeros[i]:.4f}")