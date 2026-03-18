import os
import numpy as np
import matplotlib.pyplot as plt

def generate_gue_eigenvalues(n_size):
    """
    Generates the eigenvalues of a random matrix from the Gaussian Unitary Ensemble (GUE).
    """
    # Complex Gaussian noise
    A = np.random.normal(size=(n_size, n_size)) + 1j * np.random.normal(size=(n_size, n_size))
    
    # Symmetrization to make the matrix Hermitian
    H = (A + A.conj().T) / 2
    
    # Diagonalization => eigh automatically returns sorted eigenvalues
    eigenvalues = np.linalg.eigh(H)[0]
    return eigenvalues

def unfold_gue(eigenvalues, n_size):
    """
    Unfolds the GUE eigenvalues using the Wigner Semicircle CDF.
    """
    R = 2.0 * np.sqrt(n_size)
    e_clipped = np.clip(eigenvalues, -R, R)
    cdf = 0.5 + (e_clipped / (np.pi * R)) * np.sqrt(1 - (e_clipped / R)**2) + (1.0 / np.pi) * np.arcsin(e_clipped / R)
    return n_size * cdf

def get_massive_gue_spacings(n_size=1000, nb_matrices=100):
    """
    Generates multiple GUE matrices, unfolds them, and computes the level spacings.
    """
    all_spacings = []
    print(f"Generating {nb_matrices} GUE matrices of size {n_size}x{n_size}")
    
    for i in range(nb_matrices):
        if i % 20 == 0:
            print(f"  -> Processing matrix {i}/{nb_matrices}")
        eigenvalues = generate_gue_eigenvalues(n_size)
        unfolded = unfold_gue(eigenvalues, n_size)
        spacings = np.diff(unfolded)
        all_spacings.extend(spacings)
        
    return np.array(all_spacings)

def save_gue_plots(eigenvalues, unfolded_spacings, n_size, output_dir="physics/graph"):
    """
    Generates and saves the macroscopic and microscopic plots for validation.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Global Density (Semicircle)
    plt.figure(figsize=(8, 5))
    plt.hist(eigenvalues, bins=50, density=True, color='skyblue', edgecolor='black', alpha=0.8)
    plt.title(f"Wigner Semicircle Law (N={n_size})")
    plt.savefig(os.path.join(output_dir, "wigner_semicircle.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Local Fluctuations (Surmise)
    plt.figure(figsize=(8, 5))
    plt.hist(unfolded_spacings, bins=50, range=(0, 4), density=True, color='coral', edgecolor='black', alpha=0.8)
    plt.title("Wigner Surmise: Quantum Level Repulsion")
    plt.savefig(os.path.join(output_dir, "wigner_surmise.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plots successfully saved in '{output_dir}/'")



# ==========================================
if __name__ == "__main__":
    test_n = 1000
    print(f"Testing GUE Module (N={test_n})")
    
    #  Generate one matrix for the plots
    test_eigenvalues = generate_gue_eigenvalues(test_n)
    test_spacings = np.diff(unfold_gue(test_eigenvalues, test_n))
    
    # Save the visualizations
    save_gue_plots(test_eigenvalues, test_spacings, test_n)