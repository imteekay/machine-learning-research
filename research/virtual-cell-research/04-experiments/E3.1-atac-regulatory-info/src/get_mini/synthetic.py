"""Synthetic multiome with known ground truth (positive control + test fixture).

Biology encoded on purpose:
- Each gene has a promoter peak (at the gene start) and a few distal peaks
  within +-100 kb.
- "housekeeping" genes: promoter always open, expression at a constant rate
  in every cell type (low tissue specificity, tau ~ 0).
- "enhancer" genes: expression is gated by distal enhancers whose opening
  depends on (a) the cell type and (b) a per-cell continuous "program"
  activity that is NOT a cell-type label (e.g. activation state). Programs are
  shared by a few genes, so other genes' RNA carries partial information, but
  the gene's own enhancers carry the most direct information.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .data import Multiome


def make_synthetic_multiome(
    n_cells: int = 1500, n_genes: int = 300, n_types: int = 5, frac_enhancer: float = 0.5,
    n_programs: int = 30, seed: int = 0,
) -> Multiome:
    rng = np.random.default_rng(seed)
    cell_type = rng.integers(0, n_types, n_cells)
    program = rng.normal(0, 1, (n_cells, n_programs))  # per-cell latent activity
    depth_rna = rng.lognormal(0, 0.3, n_cells)
    depth_atac = rng.lognormal(0, 0.4, n_cells)

    gene_class = np.where(rng.random(n_genes) < frac_enhancer, "enhancer", "housekeeping")
    gene_start = np.arange(n_genes) * 400_000 + 200_000
    gene_end = gene_start + rng.integers(2_000, 40_000, n_genes)

    peaks_start, peak_logit, rna_logrates = [], [], []
    for g in range(n_genes):
        base = rng.normal(0.5, 0.7)
        # promoter: open in most cells
        peaks_start.append(gene_start[g])
        peak_logit.append(np.full(n_cells, 1.0))
        n_distal = rng.integers(2, 8)
        if gene_class[g] == "housekeeping":
            for _ in range(n_distal):  # irrelevant distal peaks with random type preference
                peaks_start.append(gene_start[g] + rng.integers(-100_000, 100_000))
                peak_logit.append(rng.normal(-1.5, 0.5, n_types)[cell_type])
            rna_logrates.append(np.full(n_cells, base + 1.0))
        else:
            spec_type = rng.integers(0, n_types)
            prog = rng.integers(0, n_programs)
            drive = np.zeros(n_cells)
            for k in range(n_distal):
                peaks_start.append(gene_start[g] + rng.integers(-100_000, 100_000))
                if k < 2:  # functional enhancers: type-gated + program-driven
                    lg = -2.0 + 3.0 * (cell_type == spec_type) + 1.2 * program[:, prog]
                    drive += 1 / (1 + np.exp(-lg))
                else:
                    lg = rng.normal(-1.5, 0.5, n_types)[cell_type]
                peak_logit.append(lg)
            rna_logrates.append(base - 0.5 + 2.0 * drive)

    peak_logit = np.stack(peak_logit, 1)  # cells x peaks
    p_open = 1 / (1 + np.exp(-peak_logit))
    acc = rng.binomial(2, np.clip(p_open * np.minimum(depth_atac[:, None], 1.5) / 1.5, 0, 1))
    rates = np.exp(np.stack(rna_logrates, 1)) * depth_rna[:, None]
    rna = rng.poisson(rates)
    peaks_start = np.array(peaks_start, dtype=np.int64)
    return Multiome(
        rna=sp.csr_matrix(rna), atac=sp.csr_matrix(acc),
        gene_names=np.array([f"G{g:04d}" for g in range(n_genes)]),
        gene_chrom=np.array(["chrS"] * n_genes), gene_start=gene_start, gene_end=gene_end,
        peak_chrom=np.array(["chrS"] * len(peaks_start)), peak_start=peaks_start, peak_end=peaks_start + 500,
        barcodes=np.array([f"cell{c}" for c in range(n_cells)]),
        gene_class=gene_class, cell_type=cell_type,
    )
