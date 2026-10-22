The UCI legacy URL for the Statlog Landsat train/test files (sat.trn, sat.tst) returned 404, so analysis.py crashed before producing results.json.

Fix: analysis.py now catches the HTTP error and falls back to the OpenML copy of the same Statlog Landsat (satimage) data, using a stratified 70/30 split with seed 42. The baseline (soft-gated MoE), the improved model (class-balanced LightGBM with CV grid and sigmoid calibration), the seeds (42-44), the shared evaluation and the macro-F1 metric are unchanged. Because the split differs in the fallback case, the results describe that split and are not directly comparable to the official UCI split.

Requirements: added openml (allowed) and scipy (allowed, previously implicit). Run with `python analysis.py`; it writes results.json and figures/macro_f1_comparison.png.

## Results (latest run)

On the public UCI Statlog (Landsat Satellite) dataset, a soft-gated mixture-of-experts stand-in (not your method) reached 0.854 macro-F1 on the held-out test set, while a class-balanced LightGBM with CV grid search and sigmoid calibration reached 0.903 (single split, seeds 42-44; worst-class recall rose from 0.48 to 0.68).

| Metric | Value |
|---|---|
| baseline_score | 0.8537 |
| improved_score | 0.9031 |
| improvement | 0.04939 |
| baseline_score_std | 0.002333 |
| improved_score_std | 0.002271 |
| baseline_f1 | 0.8537 |
| improved_f1 | 0.9031 |
| baseline_accuracy | 0.8841 |
| improved_accuracy | 0.9181 |
| baseline_worst_class_recall | 0.484 |
| improved_worst_class_recall | 0.6755 |

Run on 2026-10-22. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Stand-in dataset, not the paper's LULUCF data; the paper's MoE architecture is unspecified so the baseline is a generic approximation. Test set used only once for final evaluation. Training subsampled only if above 200,000 rows (not triggered here). Results reflect this single dataset and split.

## Reference

This project is motivated by the following paper:

> SG Martín, AG Bruzón, FA Funes… - … UEx, actas de …, 2026 - revista-congresos.unex.es (2026). *Deep Learning–Based Identification of LULUCF Categories Using a Mixture of Experts Architecture and Remote Sensing Data*. revista-congresos.unex.es. https://revista-congresos.unex.es/index.php/CUEX/article/view/110

Paper author: Prof. Dr. Isaac Martín de Diego, Universidad Rey Juan Carlos (isaac.martin@urjc.es)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
