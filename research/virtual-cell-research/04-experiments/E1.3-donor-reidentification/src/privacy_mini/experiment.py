"""Linkage attack: given two disjoint cell sets from the same donors (e.g. two releases, or a public
matrix and a leaked one), can an attacker match them? And what does removing genes cost in utility?

Protocol (per seed):
1. Split donors 50/50 into DISCOVERY (attacker's / curator's reference) and EVALUATION.
2. Each evaluation donor's cells are split at random into halves A and B.
3. Pseudobulk each half (log1p CP10k of summed counts). z-score genes with discovery-donor stats.
4. Link each A to the most correlated B. Top-1 accuracy vs chance 1/N.
Two views: all cells (composition leaks too) and within one cell type (composition-controlled).
Data-driven "donor-specific" genes are ranked on DISCOVERY donors only (no leakage into evaluation).
Utility: cell-type classifier trained on discovery donors' cells, tested on evaluation donors' cells.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from .data import KNOWN_EQTL_CNV, SEX_GENES, Cohort, hla_genes


def _pb(X: sp.csr_matrix, rows: np.ndarray) -> np.ndarray:
    s = np.asarray(X[rows].sum(0)).ravel()
    return np.log1p(s / max(s.sum(), 1) * 1e4)


def pseudobulk_halves(c: Cohort, donors: np.ndarray, rng, cell_type: str | None = None, min_cells: int = 10):
    A, B, kept = [], [], []
    for d in donors:
        rows = np.where((c.donor == d) & ((c.cell_type == cell_type) if cell_type else True))[0]
        if len(rows) < 2 * min_cells:
            continue
        rows = rng.permutation(rows)
        h = len(rows) // 2
        A.append(_pb(c.X, rows[:h])); B.append(_pb(c.X, rows[h:2 * h])); kept.append(d)
    return np.array(A), np.array(B), np.array(kept)


def donor_specificity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """ICC-like score per gene: between-donor variance / within-donor (half-to-half) noise."""
    between = ((A + B) / 2).var(0)
    within = ((A - B) ** 2).mean(0) / 2
    return between / (within + 1e-6)


def link(A: np.ndarray, B: np.ndarray, mu: np.ndarray, sd: np.ndarray, gmask: np.ndarray, rng, n_boot: int = 1000) -> dict:
    a = (A[:, gmask] - mu[gmask]) / sd[gmask]
    b = (B[:, gmask] - mu[gmask]) / sd[gmask]
    a = (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-9)
    b = (b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-9)
    S = a @ b.T / a.shape[1]
    hit = (S.argmax(1) == np.arange(len(S))).astype(float)
    rank = (S > np.diag(S)[:, None]).sum(1)
    boots = hit[rng.integers(0, len(hit), (n_boot, len(hit)))].mean(1)
    return {"top1": float(hit.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "median_rank": float(np.median(rank)), "chance": 1 / len(hit), "n_donors": int(len(hit)), "n_genes": int(gmask.sum())}


def matched_random(gmask_remove: np.ndarray, mean_expr: np.ndarray, rng, n_bins: int = 20) -> np.ndarray:
    """Random gene set with the same size and expression-bin profile as the removed set."""
    bins = np.digitize(mean_expr, np.quantile(mean_expr, np.linspace(0, 1, n_bins + 1)[1:-1]))
    out = np.zeros_like(gmask_remove)
    for b in np.unique(bins[gmask_remove]):
        pool = np.where((bins == b) & ~gmask_remove)[0]
        k = int((bins[gmask_remove] == b).sum())
        out[rng.choice(pool, min(k, len(pool)), replace=False)] = True
    return out


def cell_type_f1(c: Cohort, disc: np.ndarray, ev: np.ndarray, gmask: np.ndarray, rng, n: int = 20000, min_test: int = 50) -> float | None:
    if gmask.sum() < 20:
        return None
    tr = np.where(np.isin(c.donor, disc))[0]; te = np.where(np.isin(c.donor, ev))[0]
    tr = rng.choice(tr, min(n, len(tr)), replace=False); te = rng.choice(te, min(n, len(te)), replace=False)

    def feats(rows):
        X = c.X[rows][:, gmask]
        lib = np.asarray(c.X[rows].sum(1)).ravel()
        X = sp.diags(1e4 / np.maximum(lib, 1)) @ X
        X.data = np.log1p(X.data)
        return X.toarray()

    Xtr, Xte = feats(tr), feats(te)
    pca = PCA(min(50, int(gmask.sum()) - 1), random_state=0).fit(Xtr)
    clf = LogisticRegression(max_iter=2000).fit(pca.transform(Xtr), c.cell_type[tr])
    yp, yt = clf.predict(pca.transform(Xte)), c.cell_type[te]
    types, counts = np.unique(yt, return_counts=True)
    labels = types[counts >= min_test]
    return float(f1_score(yt, yp, labels=labels, average="macro"))


def sex_accuracy(c: Cohort, A: np.ndarray, donors: np.ndarray) -> float | None:
    """Infer sex as sign(XIST - mean chrY genes) on pseudobulk; aggregate accuracy only."""
    g = list(c.genes)
    if "XIST" not in g:
        return None
    y = [g.index(s) for s in ("RPS4Y1", "DDX3Y", "UTY", "KDM5D", "EIF1AY") if s in g]
    pred = np.where(A[:, g.index("XIST")] > A[:, y].mean(1), "female", "male")
    return float((pred == c.sex[donors]).mean())


def run(c: Cohort, seed: int, ks=(50, 200, 1000), n_random: int = 5, with_utility: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    donors = rng.permutation(np.unique(c.donor))
    disc, ev = donors[: len(donors) // 2], donors[len(donors) // 2:]
    top_type = max(set(c.cell_type), key=lambda t: (c.cell_type == t).sum())
    g = c.genes
    sex = np.isin(g, SEX_GENES)
    hla = np.isin(g, hla_genes(g))
    eqtl = np.isin(g, KNOWN_EQTL_CNV)
    prereg = sex | hla | eqtl
    out: dict = {"seed": seed, "n_disc": int(len(disc)), "n_eval": int(len(ev)), "top_cell_type": top_type,
                 "set_sizes": {"sex": int(sex.sum()), "hla": int(hla.sum()), "known_eqtl_cnv": int(eqtl.sum()), "prereg": int(prereg.sum())},
                 "views": {}}
    for view, ct in (("all_cells", None), ("within_type", top_type)):
        Ad, Bd, _ = pseudobulk_halves(c, disc, rng, ct)
        Ae, Be, kept = pseudobulk_halves(c, ev, rng, ct)
        ref = np.vstack([Ad, Bd])
        mu, sd = ref.mean(0), ref.std(0) + 1e-3
        expressed = ref.mean(0) > 0.05
        spec = donor_specificity(Ad, Bd)
        spec[~expressed] = -np.inf
        order = np.argsort(-spec)
        arms = {"all_genes": expressed.copy()}
        arms["minus_sex"] = expressed & ~sex
        arms["minus_sex_hla"] = expressed & ~sex & ~hla
        arms["minus_prereg"] = expressed & ~prereg
        arms["only_prereg"] = expressed & prereg
        for k in ks:
            topk = np.zeros_like(expressed); topk[order[:k]] = True
            arms[f"minus_top{k}_specific"] = expressed & ~topk & ~prereg
        res = {name: link(Ae, Be, mu, sd, m, rng) for name, m in arms.items() if m.sum() >= 5}
        # random matched removal controls (averaged)
        mean_expr = ref.mean(0)
        for name, removed in (("prereg", expressed & prereg),
                              *[(f"top{k}_specific", expressed & np.isin(np.arange(len(g)), order[:k]) | (expressed & prereg)) for k in ks]):
            accs = [link(Ae, Be, mu, sd, expressed & ~matched_random(removed, mean_expr, rng), rng)["top1"] for _ in range(n_random)]
            res[f"minus_random_matched_{name}"] = {"top1": float(np.mean(accs)), "sd": float(np.std(accs)), "n_removed": int(removed.sum())}
        res["sex_inference_accuracy"] = sex_accuracy(c, Ae, kept) if view == "all_cells" else None
        out["views"][view] = res
        out.setdefault("top_specific_genes_discovery", {})[view] = [str(x) for x in g[order[:30]]]
    if with_utility:
        all_expr = np.asarray((c.X > 0).mean(0)).ravel() > 0.01
        out["utility_macro_f1"] = {
            "all_genes": cell_type_f1(c, disc, ev, all_expr, rng),
            "minus_prereg": cell_type_f1(c, disc, ev, all_expr & ~prereg, rng),
            "minus_prereg_top1000": cell_type_f1(c, disc, ev, all_expr & ~prereg & ~np.isin(np.arange(len(g)), order[:1000]), rng),
        }
    return out
