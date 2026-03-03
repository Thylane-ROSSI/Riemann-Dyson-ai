import os
import numpy as np
import matplotlib.pyplot as plt

def generate_gue_eigenvalues(n_size):
    """
    Generates the eigenvalues of a random matrix from the Gaussian Unitary Ensemble (GUE).
    
    Parameters:
    n_size (int): Dimension of the GUE matrix.
    
    Returns:
    numpy.ndarray: Sorted array of real eigenvalues.
    """
    # Complex Gaussian noise
    A = np.random.normal(size=(n_size, n_size)) + 1j * np.random.normal(size=(n_size, n_size))
    
    # Symmetrization to make the matrix Hermitian
    H = (A + A.conj().T) / 2
    
    # Diagonalization (eigh automatically returns sorted eigenvalues)
    eigenvalues = np.linalg.eigh(H)[0]
    
    return eigenvalues

def unfold_gue(eigenvalues, n_size):
    """
    Unfolds the GUE eigenvalues using the theoretical Wigner Semicircle Cumulative 
    Distribution Function (CDF) to normalize the mean local spacing to 1.
    
    Parameters:
    eigenvalues (numpy.ndarray): Sorted array of raw eigenvalues.
    n_size (int): Dimension of the GUE matrix.
    
    Returns:
    numpy.ndarray: Unfolded eigenvalues.
    """
    # Theoretical radius of the semicircle
    R = 2.0 * np.sqrt(n_size)
    
    # Clip eigenvalues to [-R, R] to prevent domain errors in sqrt and arcsin
    # due to slight statistical fluctuations at the spectral edges
    e_clipped = np.clip(eigenvalues, -R, R)
    
    # Wigner Semicircle CDF integration
    cdf = 0.5 + (e_clipped / (np.pi * R)) * np.sqrt(1 - (e_clipped / R)**2) + (1.0 / np.pi) * np.arcsin(e_clipped / R)
    
    # Scale to [0, n_size] to achieve a mean level spacing of 1
    unfolded_eigenvalues = n_size * cdf
    
    return unfolded_eigenvalues

if __name__ == "__main__":
    # Ensure the output directory exists
    output_dir = "graph"
    os.makedirs(output_dir, exist_ok=True)
    
    n_size = 1000
    print(f"Generating GUE matrix (N={n_size})...")
    
    # Raw Data Generation
    eigenvalues = generate_gue_eigenvalues(n_size)
    print(f"Successfully generated {n_size} eigenvalues. Lowest energy level: {eigenvalues[0]:.2f}")
    
    # Global Density Visualization
    plt.figure(figsize=(8, 5))
    plt.hist(eigenvalues, bins=50, density=True, color='skyblue', edgecolor='black', alpha=0.8)
    plt.title(f"Wigner Semicircle Law (N={n_size})")
    plt.xlabel("Energy Level (Eigenvalue)")
    plt.ylabel("Density")
    
    semicircle_path = os.path.join(output_dir, "wigner_semicircle.png")
    plt.savefig(semicircle_path, dpi=300, bbox_inches='tight')
    print(f"Global density plot saved to '{semicircle_path}'")
    plt.close()
    
    # Unfolding Process
    unfolded_eigenvalues = unfold_gue(eigenvalues, n_size)
    unfolded_spacings = np.diff(unfolded_eigenvalues)
    
    mean_spacing = np.mean(unfolded_spacings)
    print(f"Mean spacing AFTER unfolding: {mean_spacing:.4f} (Theoretical = 1.0000)")
    
    # Local Fluctuations Visualization
    plt.figure(figsize=(8, 5))
    plt.hist(unfolded_spacings, bins=50, range=(0, 4), density=True, color='coral', edgecolor='black', alpha=0.8)
    plt.title("Wigner Surmise: Quantum Level Repulsion")
    plt.xlabel("Spacing (s) between adjacent energy levels")
    plt.ylabel("Probability Density")
    
    surmise_path = os.path.join(output_dir, "wigner_surmise.png")
    plt.savefig(surmise_path, dpi=300, bbox_inches='tight')
    print(f"Level repulsion plot saved to '{surmise_path}'")
    plt.close()