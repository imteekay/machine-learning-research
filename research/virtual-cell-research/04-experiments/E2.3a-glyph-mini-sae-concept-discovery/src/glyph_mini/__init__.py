"""glyph-mini: a minimal, educational version of the Glyph pipeline.

Stages (mirroring the paper):
    synthetic   -> stand-in for frozen foundation-model embeddings with known ground truth
    sae / train -> Matryoshka Top-K sparse autoencoder ("concept dictionary")
    association -> concept x outcome logistic regression, Bonferroni / BH
    dml         -> double/debiased ML confounder adjustment
    pipeline    -> discovery on site A, replication on site B, adjustment
"""
