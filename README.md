# Class-Balanced LightGBM vs Mixture-of-Experts Baseline for Land-Cover Classification

Land-use and land-use change categories (LULUCF) are increasingly identified from satellite imagery, and the paper "Deep Learning–Based Identification of LULUCF Categories Using a Mixture of Experts Architecture and Remote Sensing Data" (2026) proposes a mixture-of-experts network for this task. This experiment asks whether a simpler, class-balanced, cross-validated gradient-boosting model with calibrated probabilities can match or beat a standard soft-gated mixture-of-experts stand-in on multispectral pixel data.

**Question:** Can a class-balanced, cross-validated gradient-boosting model with per-band attribution-ready features beat a standard mixture-of-experts-style baseline at classifying land cover categories from multispectral pixel data?

**Dataset:** UCI Statlog (Landsat Satellite) — 6 spectral bands from 3x3 pixel neighbourhoods, 6 land-cover classes, official train/test split. This is a public stand-in dataset, not the paper's own LULUCF data.

**Baseline:** soft-gated mixture of four small MLP experts with a logistic-regression gating network.

**Improved model:** class-balanced LightGBM with a small cross-validated grid search on the training set and sigmoid probability calibration, using the same standardised inputs.

**How to run**

```
pip install -r requirements.txt
python analysis.py
```

**Results:** results are written to `results.json` (and a chart to `figures/`).

## Reference

This project is motivated by the following paper:

> SG Martín, AG Bruzón, FA Funes… - … UEx, actas de …, 2026 - revista-congresos.unex.es (2026). *Deep Learning–Based Identification of LULUCF Categories Using a Mixture of Experts Architecture and Remote Sensing Data*. revista-congresos.unex.es. https://revista-congresos.unex.es/index.php/CUEX/article/view/110

Paper author: Prof. Dr. Isaac Martín de Diego, Universidad Rey Juan Carlos (isaac.martin@urjc.es)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
