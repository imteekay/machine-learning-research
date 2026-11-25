"""perturbseq-mini: a minimal Perturb-seq PoC (Dixit et al. 2016) for experiment E0.2.

Modules
-------
io          load the Dixit 2016 K562 TF screen from GEO files, and CollecTRI
synthetic   GRN-based synthetic Perturb-seq with known ground truth (tests / fallback)
preprocess  QC metrics, gene filter, normalization, guide -> perturbation design matrix
mimosca     MIMOSCA-style ridge regression of expression on guide indicators + covariates
enrichment  TF target-set test vs expression-matched random sets, permutation p + BH FDR
"""
