# Quantum vs. Classical Machine Learning Benchmark

Comparative benchmark of a **Variational Quantum Classifier (VQC)**, a **Quantum Support Vector Machine (QSVM)**, and a **Classical SVM (RBF kernel)** across four datasets of increasing difficulty — implemented with [PennyLane](https://pennylane.ai/) (`lightning.qubit` simulator) and scikit-learn.

This is the experimental codebase for the senior project *"Comparative Analysis of Quantum Machine Learning Architectures Against Classical Methods: VQC, QSVM, and Classical SVM"* (Erzurum Technical University, Department of Computer Engineering).

## What this project does

- Trains and evaluates **VQC**, **QSVM**, and **Classical SVM** on four datasets: Iris, Wine, Breast Cancer, and the deliberately noisy Madelon dataset.
- Compares two quantum data encoding strategies: **Angle Encoding** vs. **Amplitude Encoding**.
- Tests **qubit scaling** (3 vs. 4 qubits) to observe Barren Plateau effects.
- Validates results with both a **Train/Test split** and **2-Fold Cross-Validation**.
- Reports accuracy and wall-clock training time for every configuration, and writes the full result table to a CSV file.

## Repository structure

```
├── README.md
├── report.pdf          # Full thesis report
├── requirements.txt
├── src/                # Source code
│   ├── main.py             # Benchmark on Iris, Wine, Breast Cancer
│   └── madelon_test.py     # Benchmark on Madelon (noise robustness)
├── docs/               # Extended methodology notes
│   └── METHODOLOGY.md
├── simulations/        # Console output screenshots from real runs
├── results/            # CSV result tables (per-run and summary)
└── presentation/        # Project presentation slides
```

## Files

| File | Description |
|---|---|
| `src/main.py` | Runs the full benchmark on Iris, Wine, and Breast Cancer. Outputs `qml_benchmark_lightning.csv`. |
| `src/madelon_test.py` | Runs the same benchmark on the Madelon dataset only (noise-robustness stress test). Outputs `qml_benchmark_madelon_only.csv`. |

## How to run

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. (Recommended) create a virtual environment
python -m venv qml_env
qml_env\Scripts\activate      # Windows
source qml_env/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the benchmarks
python src/main.py
python src/madelon_test.py
```

Each script prints live progress to the console (see `simulations/` for example output) and saves a results table as a `.csv` file in the project's root folder (curated copies are versioned in `results/`).

## Results

### Iris, Wine, Breast Cancer (`main.py`)

**Overall model comparison (averaged across all datasets, encodings, and qubit counts):**

| Model | Accuracy | Avg. Time |
|---|---|---|
| Classical SVM (RBF) | 0.9870 | 0.00 s |
| QSVM | 0.8837 | 50.29 s |
| VQC | 0.7615 | 1.99 s |

**Angle vs. Amplitude encoding (QML models only):**

| Encoding | Accuracy | Avg. Time |
|---|---|---|
| Angle | 0.9433 | 9.38 s |
| Amplitude | 0.7019 | 42.90 s |

**Qubit scaling (QML models only):**

| Qubits | Accuracy | Avg. Time |
|---|---|---|
| 3 | 0.8306 | 21.33 s |
| 4 | 0.8145 | 30.95 s |

**Validation method:**

| Validation | Accuracy | Avg. Time |
|---|---|---|
| Train/Test Split | 0.8469 | 17.41 s |
| 2-Fold CV | 0.8348 | 29.07 s |

### Madelon — noise robustness test (`madelon_test.py`)

Madelon has 500 features, 480 of which are irrelevant/noise. Random-chance accuracy is 50%.

| Model | Accuracy | Avg. Time |
|---|---|---|
| QSVM | 0.5133 | 53.85 s |
| VQC | 0.5092 | 2.13 s |
| Classical SVM (RBF) | 0.3956 | 0.00 s |

**Key finding:** on clean, well-structured data, Classical SVM wins decisively. But on noisy, high-dimensional data (Madelon), Classical SVM collapses *below* random chance (39.6%), while both quantum models stay *above* random chance — a concrete, reproducible instance of quantum robustness to feature noise.

Full per-dataset, per-configuration numbers are available in [`results/qml_benchmark_lightning.csv`](results/qml_benchmark_lightning.csv) and [`results/qml_benchmark_madelon_only.csv`](results/qml_benchmark_madelon_only.csv).

## Method summary

- **Quantum backend:** PennyLane `lightning.qubit` (C++-accelerated CPU simulator).
- **Encoding:** Angle Encoding (`qml.AngleEmbedding`, PCA → `[0, π]` scaling) and Amplitude Encoding (`qml.AmplitudeEmbedding`, PCA/padding → L2-normalized).
- **VQC:** `StronglyEntanglingLayers` ansatz (2 layers), trained with the Adam optimizer via `diff_method="adjoint"`.
- **QSVM:** quantum kernel computed as the |⟨0|U†(x₂)U(x₁)|0⟩|² overlap probability, fed into scikit-learn's `SVC(kernel="precomputed")`.
- **Classical baseline:** scikit-learn `SVC(kernel="rbf")`.
- All features are standardized (`StandardScaler`) before model-specific preprocessing.

## Limitations

- All experiments run on an exact simulator — no real quantum hardware noise is present.
- Sample sizes are capped (150 examples) to keep quantum simulation time practical.
- Results are single-run per configuration except where 2-Fold CV is used.

## More

- Full thesis write-up: [`report.pdf`](report.pdf)
- Extended methodology notes: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)
- Raw console output from real runs: [`simulations/`](simulations/)
- Presentation slides: [`presentation/`](presentation/)

## License

Educational/academic use — no license specified. Add one (e.g. MIT) if you plan to share this more broadly.
