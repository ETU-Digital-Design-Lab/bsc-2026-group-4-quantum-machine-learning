# Methodology

Condensed technical summary of the experimental setup. Full details, background, and discussion are in `../report.pdf` (Chapters 3–4).

## Datasets

| Dataset | Samples | Features | Classes | Role |
|---|---|---|---|---|
| Iris | 150 | 4 | 3 (species) | Simple sanity check |
| Wine | 178 | 13 | 3 (origin) | Moderate complexity |
| Breast Cancer | 569 | 30 | 2 (B/M) | Higher-dimensional, medical data |
| Madelon | 2600 | 500 | 2 (random) | 480/500 features are noise — stress test for noise robustness |

## Preprocessing

- All features standardized with `StandardScaler` (zero mean, unit variance).
- For quantum models, PCA reduces dimensionality to match the qubit count (Angle Encoding) or to `2^n_qubits` (Amplitude Encoding, with zero-padding if needed).
- Angle Encoding features are scaled to `[0, π]`; Amplitude Encoding vectors are L2-normalized.
- Stratified train/test splits; results also validated with 2-Fold Stratified Cross-Validation.

## Models

**Classical SVM** — scikit-learn `SVC(kernel="rbf")`, default hyperparameters.

**Variational Quantum Classifier (VQC)**
- Backend: PennyLane `lightning.qubit`.
- Ansatz: `StronglyEntanglingLayers` (2 layers).
- Optimizer: Adam (`stepsize=0.1`), mini-batch training, `diff_method="adjoint"`.
- Output: expectation value of Pauli-Z on qubit 0, thresholded at 0.

**Quantum Support Vector Machine (QSVM)**
- Quantum kernel: overlap probability `|⟨0|U†(x₂)U(x₁)|0⟩|²`, computed via a PennyLane circuit for every training/test pair.
- The precomputed kernel matrix is passed to scikit-learn `SVC(kernel="precomputed")`.
- No quantum parameters are trained — only the classical SVM stage is optimized.

## Data encoding strategies compared

- **Angle Encoding** (`qml.AngleEmbedding`): one feature per qubit, shallow circuit, needs `n_qubits = n_features`.
- **Amplitude Encoding** (`qml.AmplitudeEmbedding`): features packed into `2^n_qubits` state amplitudes, needs fewer qubits but a deeper state-preparation circuit.

## Experiment grid

Every dataset × model × encoding × qubit-count (3, 4) × validation method (Train/Test split, 2-Fold CV) combination is run and logged (Classical SVM is encoding/qubit-agnostic, so it only runs once per dataset/validation method). See `../results/` for the full grid and `../simulations/` for the raw console logs of these runs.
