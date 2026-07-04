# Simulations

Console output captured while running the benchmark scripts in `src/`, kept as visual evidence of real experimental runs (in addition to the numeric results in `results/`).

| File | From | Description |
|---|---|---|
| `main_benchmark_run.png` | `python src/main.py` | Live per-configuration output for Iris, Wine, and Breast Cancer (VQC / QSVM / Classical SVM, both encodings, 3 and 4 qubits, both validation methods). |
| `main_benchmark_summary.png` | `python src/main.py` | Final aggregated analysis report printed at the end of the run. |
| `madelon_benchmark_run.png` | `python src/madelon_test.py` | Live per-configuration output and final analysis report for the Madelon noise-robustness test. |

The corresponding machine-readable numbers are in `../results/`.
