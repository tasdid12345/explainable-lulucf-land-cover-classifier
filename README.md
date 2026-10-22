The UCI legacy URL for the Statlog Landsat train/test files (sat.trn, sat.tst) returned 404, so analysis.py crashed before producing results.json.

Fix: analysis.py now catches the HTTP error and falls back to the OpenML copy of the same Statlog Landsat (satimage) data, using a stratified 70/30 split with seed 42. The baseline (soft-gated MoE), the improved model (class-balanced LightGBM with CV grid and sigmoid calibration), the seeds (42-44), the shared evaluation and the macro-F1 metric are unchanged. Because the split differs in the fallback case, the results describe that split and are not directly comparable to the official UCI split.

Requirements: added openml (allowed) and scipy (allowed, previously implicit). Run with `python analysis.py`; it writes results.json and figures/macro_f1_comparison.png.

## Reference

This project is motivated by the following paper:

> SG Martín, AG Bruzón, FA Funes… - … UEx, actas de …, 2026 - revista-congresos.unex.es (2026). *Deep Learning–Based Identification of LULUCF Categories Using a Mixture of Experts Architecture and Remote Sensing Data*. revista-congresos.unex.es. https://revista-congresos.unex.es/index.php/CUEX/article/view/110

Paper author: Prof. Dr. Isaac Martín de Diego, Universidad Rey Juan Carlos (isaac.martin@urjc.es)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
