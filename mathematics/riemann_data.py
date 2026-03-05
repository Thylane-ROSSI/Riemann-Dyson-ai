import os
import numpy as np
import matplotlib.pyplot as plt

def load_riemann_zeros(filepath):
    """
    Loads the Riemann Zeta zeros from a text file.
    """
    print(f"Loading data from {filepath}...")
    try:
        zeros = np.loadtxt(filepath)
        return zeros
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        print("Please make sure you downloaded it and put it in the 'data' folder.")
        return None

def unfold_riemann_zeros(gamma_array):
    """
    Unfolds the Riemann zeros using the leading asymptotic term of the counting function N(T).
    """
    unfolded = (gamma_array / (2 * np.pi)) * (np.log(gamma_array / (2 * np.pi)) - 1)
    return unfolded


if __name__ == "__main__":
    # Define the path to our downloaded file (remonter d'un dossier depuis 'mathematics')
    data_path = os.path.join("../data", "zeros_100k.txt")
    
    # Load the zeros
    riemann_zeros = load_riemann_zeros(data_path)
    
    if riemann_zeros is not None:
        num_zeros = len(riemann_zeros)
        print(f"\nSuccess! Loaded {num_zeros} Riemann zeros.")
        
        # Physics / Math check: The first zero should be around 14.1347
        print(f"First zero: {riemann_zeros[0]:.4f}")

        # === L'UNFOLDING ===
        unfolded_zeros = unfold_riemann_zeros(riemann_zeros)
        unfolded_spacings = np.diff(unfolded_zeros)
        
        mean_spacing = np.mean(unfolded_spacings)
        print(f"\nMean spacing of the unfolded zeros: {mean_spacing:.4f} (Theoretical = 1.0000)")

        # === AFFICHAGE ===
        output_dir = "graph" # On pointe vers le dossier global
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(8, 5))
        plt.hist(unfolded_spacings, bins=50, range=(0, 4), density=True, color='mediumseagreen', edgecolor='black', alpha=0.8)
        
        # Titres adaptés aux mathématiques
        plt.title("Riemann Zeros: Spacing Distribution")
        plt.xlabel("Spacing (s) between adjacent zeros")
        plt.ylabel("Probability Density")
        
        surmise_path = os.path.join(output_dir, "riemann_surmise.png")
        plt.savefig(surmise_path, dpi=300, bbox_inches='tight')
        print(f"Level repulsion plot saved to '{surmise_path}'")
        plt.close()