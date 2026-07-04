# Results

CSV result tables from the benchmark runs.

| File | Description |
|---|---|
| `qml_benchmark_lightning.csv` | Full per-configuration results for `src/main.py` (Iris, Wine, Breast Cancer): every combination of model, encoding, qubit count, and validation method, with accuracy and training time. |
| `qml_benchmark_madelon_only.csv` | Same, for `src/madelon_test.py` (Madelon dataset only). |
| `main_benchmark_summary.csv` | Aggregated summary for the Iris/Wine/Breast Cancer run: mean accuracy and time grouped by validation method, model, encoding, and qubit count. |
| `madelon_benchmark_summary.csv` | Same aggregation, for the Madelon run. |

Re-running `src/main.py` or `src/madelon_test.py` regenerates the full per-configuration CSVs (`qml_benchmark_lightning.csv` / `qml_benchmark_madelon_only.csv`) at the project root. The copies here are the exact output files from the experiments reported in `report.pdf`.
