import os
import numpy as np
import matplotlib.pyplot as plt

def load_riemann_zeros(filepath):
    """
    Loads the raw Riemann Zeta zeros (gamma heights) from a text file.
    """
    print(f"Loading Riemann data from {filepath}")
    try:
        return np.loadtxt(filepath)
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}.")
        return None
        
def unfold_riemann(gamma_array):
    """
    Unfolds the Riemann zeros to normalize the local density to 1.
    """
    unfolded = (gamma_array / (2 * np.pi)) * (np.log(gamma_array / (2 * np.pi)) - 1)
    return unfolded

def get_riemann_spacings(filepath):
    """
    Full pipeline to load, unfold, and return the spacings of Riemann zeros.
    """
    zeros = load_riemann_zeros(filepath)
    if zeros is not None:
        unfolded_zeros = unfold_riemann(zeros)
        return np.diff(unfolded_zeros)
    return None

def save_riemann_plot(unfolded_spacings, output_dir="../mathematics"):
    """
    Saves the spacing distribution plot to visually verify the level repulsion.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(unfolded_spacings, bins=50, range=(0, 4), density=True, color='mediumseagreen', edgecolor='black', alpha=0.8)
    plt.title("Riemann Zeros: Spacing Distribution")
    plt.savefig(os.path.join(output_dir, "riemann_surmise.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot successfully saved in '{output_dir}/'")


# ==========================================
if __name__ == "__main__":
    print("--- Testing Riemann Module ---")
    
    test_data_path = os.path.join("../data", "zeros_100k.txt")
    
    # Get the spacings directly using our new helper function
    spacings = get_riemann_spacings(test_data_path)
    
    if spacings is not None:
        print(f"Mean spacing: {np.mean(spacings):.4f} (Theoretical = 1.0000)")
        save_riemann_plot(spacings, output_dir="graph")