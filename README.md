# The Riemann-Dyson AI Oracle

This repository contains a research project bridging **Random Matrix Theory**, **Number Theory** (Riemann Zeta function), and **Deep Learning**. 

The ultimate goal of this project is to train a neural network to distinguish between quantum chaos (energy levels of heavy nuclei) and mathematical chaos (zeros of the Riemann Zeta function), and to use **Explainable AI** to interpret the model's decision boundaries.

## Theoretical Background

In the 1970s, physicist Freeman Dyson and mathematician Hugh Montgomery made a discovery at Princeton: the equations describing the energy levels of complex, heavy atoms (Quantum Chaos) perfectly matched the equations describing the distribution of prime numbers (Riemann Zeta function). 

* **The Physical Side (GUE):** Heavy atoms like Uranium are too complex to model exactly. Instead, physicists use the **Gaussian Unitary Ensemble (GUE)**—random matrices filled with complex noise—to simulate their Hamiltonians. The eigenvalues of these matrices represent the energy levels.
* **The Mathematical Side (Riemann):** The non-trivial zeros of the Riemann Zeta function dictate the distribution of prime numbers.
* **The Conjecture:** The spacing between Riemann zeros statistically mimics the spacing between quantum energy levels (they both exhibit "level repulsion").

**The AI Challenge:** If the statistics are theoretically identical, can a modern Deep Learning model find a hidden pattern to tell them apart? And if so, *how*?

---

## Phase 1 (Building the Physics Baseline)

Before feeding data to an AI, we must generate and standardize the physical dataset. Phase 1 focuses on simulating heavy nuclei and validating the quantum mechanical properties.

### How Phase 1 Works:
1. **Matrix Generation:** We generate large random Hermitian matrices (N=1000) using complex Gaussian noise to simulate chaotic quantum systems.
2. **Diagonalization:** We extract the eigenvalues (the raw energy levels) from these matrices. Plotted globally, they form the famous **Wigner Semicircle Law**.
3. **The Unfolding Process:** Raw energy levels are denser in the middle of the spectrum and sparse at the edges. To study local fluctuations (the spacing between adjacent levels), we must "unfold" the spectrum. We apply a non-linear transformation using the theoretical Wigner Cumulative Distribution Function (CDF) to normalize the mean level spacing to exactly 1.0 everywhere.
4. **Validation:** We compute the distances between these unfolded levels. The resulting distribution perfectly matches the **Wigner Surmise**, visually demonstrating "Quantum Level Repulsion" (energy levels refuse to overlap).

### Visualizations (Phase 1)

**1. Global Density: The Wigner Semicircle Law** *The macroscopic distribution of raw energy levels.* ![Wigner Semicircle](physics/graph/wigner_semicircle.png)

**2. Local Fluctuations: The Wigner Surmise** *The microscopic spacing after the unfolding process. Notice how the probability drops to zero on the left: energy levels repel each other.* ![Wigner Surmise](physics/graph/wigner_surmise.png)

---

## Phase 2 (The Mathematical Baseline: Riemann Zeta Function)

Just like the energy levels of heavy atoms, the non-trivial zeros of the Riemann Zeta function exhibit varying global density (they become denser as we move higher up the critical line). To compare them with our quantum system, we must apply a similar unfolding process.

### How Phase 2 Works:
1. **Data Acquisition:** We utilize the high-precision datasets computed by mathematician Andrew Odlyzko, loading the heights ($\gamma$) of the first 100,000 non-trivial zeros.
2. **The Unfolding Process:** We unfold the Riemann zeros using the leading asymptotic term of the Riemann-von Mangoldt counting function N(T):
   $$x_i = \frac{\gamma_i}{2\pi} \left( \ln\left(\frac{\gamma_i}{2\pi}\right) - 1 \right)$$
   This transformation normalizes the local density of the zeros, forcing the mean spacing between adjacent zeros to be exactly 1.0.
3. **The Conjecture:** When we plot the histogram of the spacings between these unfolded zeros, we witness the exact same "Quantum Level Repulsion" (Wigner Surmise) as seen in the GUE matrices. The mathematical chaos perfectly mirrors the quantum chaos.

### Visualizations (Phase 2)

**Riemann Zeros Local Fluctuations** *The microscopic spacing of the unfolded Riemann zeros. The curve is virtually identical to the GUE distribution.*
![Riemann Surmise](mathematics/graph/riemann_surmise.png)

---

## Phase 3 (The Deep Learning Discriminator)

While the histograms of GUE and Riemann spacings are visually identical, Phase 3 aims to determine if a neural network can find hidden, higher-order correlations within sequences of these spacings to distinguish between the two systems.

### Architecture: Multi-Layer Perceptron (MLP)
We built a custom neural network using PyTorch. The model takes a sequence of 50 consecutive unfolded spacings as input and outputs the probability of the sequence belonging to the Riemann Zeta function (Label 1) versus a GUE matrix (Label 0).

The architecture consists of approximately 5,300 trainable parameters distributed across three fully connected linear layers:
* **Input:** A sliding window of 50 consecutive level spacings.
* **Hidden Layer 1:** 64 neurons with a ReLU activation function to introduce non-linearity.
* **Hidden Layer 2:** 32 neurons with a ReLU activation function.
* **Output Layer:** 1 neuron with a Sigmoid activation function to compress the final mathematical score into a binary probability.

### Training Methodology
* **Dataset Generation:** We generated ~100,000 GUE spacings from 100 different matrices (N=1000) to ensure high physical variance, and combined them with the first 100,000 unfolded Riemann zero spacings. 
* **Data Split:** The dataset of approximately 200,000 sequences was strictly divided into an 80% training set and a 20% test set to evaluate true generalization and prevent overfitting.
* **Optimization:** The model was trained using the Adam optimizer (Learning Rate = 0.001) and Binary Cross Entropy (BCE) as the loss function. The training ran for 5 epochs with a batch size of 32.

### Results & Interpretation
The neural network achieved remarkable results:
* **Training Accuracy:** ~97.45%
* **Test Accuracy:** **97.62%**

This exceptionally high accuracy on unseen test data formally proves that the model successfully generalized. It confirms that the sequence of distances between Riemann zeros contains a distinct mathematical signature separate from pure quantum randomness. While their nearest-neighbor distributions (Wigner Surmise) align perfectly to the human eye and basic statistics, the MLP successfully mapped higher-order correlations across the 50-step sequences. The trained weights are saved in a `.pth` file for future inference.

---

## Phase 4 (Explainable AI and Interpretability)

While our neural network accurately distinguishes between GUE and Riemann spacings, it effectively acts as a "black box." Phase 4 aims to look inside this black box using **Explainable AI (XAI)** to understand *how* the model makes its decisions and which parts of the sequences carry the distinct mathematical signatures.

### How Phase 4 Works:
1. **Game Theory Approach:** We use **SHAP (SHapley Additive exPlanations)**, a method rooted in cooperative game theory. SHAP assigns an importance value to each feature (in our case, each of the 50 level spacings in a sequence) based on its contribution to the model's final prediction.
2. **DeepExplainer Implementation:** We utilize `shap.DeepExplainer` to approximate Shapley values for our PyTorch MLP. We feed the explainer a "background dataset" (a sample of sequences to establish a baseline) and a test dataset to analyze.
3. **Dimensionality Wrapper:** To ensure compatibility between PyTorch's squeezed output layers and SHAP's tensor expectations, the model is wrapped in a custom `nn.Module` that re-expands the output to `[batch_size, 1]`.

### Results & Interpretation
The SHAP values allow us to generate a summary plot that ranks the features by their impact on the model's output. By analyzing this graph, we can observe:
* **Feature Importance:** Which specific spacings within the 50-step window (e.g., the early spacings vs. the later ones) are most heavily weighted by the model.
* **Correlation Direction:** Whether a higher or lower value in a specific spacing pushes the model's prediction toward the Riemann Zeta function (Label 1) or toward the GUE distribution (Label 0).

This phase effectively translates the abstract weights of the neural network back into readable mathematical insights, providing clues about the structural differences between quantum random matrices and prime number distribution.

---

## Phase 5 (Global Interpretability: Decoding the AI's Logic)

After demonstrating that our model could distinguish GUE and Riemann with 97.6% accuracy (Phase 3) and identifying influential spacings for individual predictions (Phase 4), this final phase aims to understand globally *how* the model makes its decisions. 

To achieve this, we scaled our SHAP analysis to aggregate the impact of each spacing across the entire dataset (5,000 sequences), revealing the underlying structural motifs learned by the neural network.

### Key Results

**1. Central Spacings are the Most Discriminative**
The model largely ignores the first and last spacings of a sequence (positions 1–17 and 40–50) and focuses heavily on a central window (positions 18–35). This suggests that:
* Medium-range correlations carry a distinct mathematical signature that separates GUE and Riemann.
* Very short-range spacings (1–7) might be too similar between the two systems (as they are both dominated by immediate "level repulsion"), while long-range spacings (40–50) might simply be too noisy.

**2. The Model Associates Large Spacings with Riemann**
The global SHAP analysis reveals a clear directional pattern:
* High values for certain central spacings (e.g., 21, 22, 31) strongly push the prediction toward "Riemann".
* Low values for these same spacings push the prediction toward "GUE".
* *Hypothesis:* The zeros of the Riemann Zeta function might exhibit a higher spacing variance than GUE energy levels within these specific correlation windows. This hypothesis remains to be statistically validated.

**3. Symmetry and Recurring Motifs**
Certain adjacent spacings (e.g., 19/20 or 32/33) share almost identical importance weights. This suggests the model is exploiting pairs or triplets of spacings to capture local motifs, reminiscent of N-body correlations in quantum physics.

### Visualizations (Phase 5)

**1. Global SHAP Summary (Bar Plot)**
*The top 20 most influential spacings on the model's output, ranked by average absolute importance. The central cluster (spacings 18–35) clearly dominates the decision-making process.*
![Global SHAP Summary](shap_global_summary.png)

---

## Limitations & Epistemological Biases

Despite these highly promising results, several limitations must be rigorously highlighted:

**1. Data Selection Bias**
* **GUE:** Generated from fixed-size matrices (N=1000) using only 100 realizations. The model might have captured numerical artifacts specific to this matrix dimension.
* **Riemann:** We utilized the first 100,000 zeros. The analytical unfolding process is inherently less precise at the beginning of the critical line than it is for much higher zeros.

**2. Imperfect Unfolding**
The unfolding of the Riemann zeros relies on an asymptotic approximation (the leading term of the Riemann-von Mangoldt formula). Because this formula is only perfectly accurate as T approaches infinity, the earliest zeros in our dataset might be imperfectly normalized, potentially leaving behind a density artifact that the AI exploited.

**3. Potential Overfitting to the Distribution**
A 97.6% accuracy is suspiciously high for a task historically deemed impossible by traditional statistics. Further testing is strictly necessary to verify true generalization (e.g., evaluating the model on GUE matrices generated via different algorithms, or testing on Riemann zeros located around $10^{20}$).

**4. Limits of Explainability (SHAP Assumption)**
SHAP mathematically assumes that features are independent. However, in a sequence of spacings, features are inherently correlated. Consequently, the SHAP values might slightly overestimate or misattribute the importance of certain specific spacings.

---

## Conclusion & Future Perspectives

### What We Learned
* An Artificial Intelligence can indeed distinguish GUE from Riemann despite their theoretical microscopic similarity, primarily by exploiting medium-range correlations.
* The central spacings within our sliding window (18–35) appear to carry a distinct mathematical signature, potentially linked to N-body correlations (where N > 2) that differ from quantum randomness.
* The unfolding process is the critical foundation of this research: without it, the model would simply capture trivial global density differences rather than true local chaotic fluctuations.

### Takeaways with Caution
* This is **not** a formal mathematical proof. It is an empirical demonstration within a highly specific computational framework.
* Biases are prevalent: numerical artifacts, unfolding approximations, and potential dataset overfitting must be considered.
* Generalization remains uncertain until tested against vastly different datasets.

---

## Project Roadmap

- [x] **Phase 1:** Simulate GUE matrices, implement analytical unfolding, and validate quantum repulsion.
- [x] **Phase 2:** Acquire and unfold the non-trivial zeros of the Riemann Zeta function (via Odlyzko's datasets).
- [x] **Phase 3:** Design and train a Deep Learning discriminator (MLP) to classify GUE vs. Riemann sequences with >97% accuracy.
- [x] **Phase 4:** Apply Explainable AI (SHAP/Captum) to interpret the model's logic and extract the mathematical boundary discovered by the neural network.
- [x] **Phase 5:** Scale SHAP analysis to the global dataset (5,000 sequences) to decode the AI's logic, isolate central medium-range correlations, and acknowledge epistemological limitations.

## Requirements & Usage
* Python 3.x
* `numpy`, `matplotlib`, `torch`

To generate a GUE matrix (N=1000), perform the unfolding, and generate the plots in the local `graph/` directory:
```bash
python3 -m physics.gue_generator
```

To load the zeros of the Riemann Zeta function, perform the unfolding, and generate the plot in the local graph/ directory:

``` bash
python3 -m mathematics.riemann_data
```

To execute the data pipeline and train the Multi-Layer Perceptron from scratch:

```bash
python3 -m ml.train
```
To run the Explainable AI (SHAP) global analysis and generate the interpretability plots in the local graph/ directory:

```bash
python3 -m ml.explain
```

## References & Documentation

* **[Granville, A. (2002). Nombres premiers et chaos quantique.](https://dms.umontreal.ca/~andrew/PDF/quantique.pdf)** Details the historical context and the surprising connection between prime number distribution and the mathematics of quantum physics.
* **[Crouzet, A. (2010). Introduction aux Matrices Aléatoires.](https://www.math.ens.psl.eu/shared-files/10508/?crouzet.pdf)** Provides a rigorous introduction to Random Matrix Theory and the Gaussian Unitary Ensemble (GUE).
* **[Exo7. Valeurs propres, vecteurs propres.](http://exo7.emath.fr/cours/ch_vp.pdf)** Fundamental linear algebra course on eigenvalues and matrix diagonalization.
* **Derbyshire, J. (2004).** *Prime Obsession: Bernhard Riemann and the Greatest Unsolved Problem in Mathematics.* (Dans la jungle des nombres premiers).