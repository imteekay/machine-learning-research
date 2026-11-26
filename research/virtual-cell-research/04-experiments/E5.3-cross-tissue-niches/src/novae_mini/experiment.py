"""Leave-one-tissue-out niche transfer. Niche embeddings are scored by how well unsupervised clusters of a held-out tissue
recover its fine anatomical regions (CCF), with no label of that tissue used anywhere."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .data import Brain
from .models import GenePca, embed, shuffled_graph, train


def eval_set(b: Brain, tissue: str, n_eval: int = 40_000, min_niche: int = 1000, max_k: int = 25, seed: int = 0):
    """Cells of the tissue that belong to niches with enough cells; K = number of those niches (capped)."""
    idx = np.where(b.tissue == tissue)[0]
    niches, counts = np.unique(b.niche[idx], return_counts=True)
    keep = niches[counts >= min_niche]
    idx = idx[np.isin(b.niche[idx], keep)]
    if len(idx) > n_eval:
        idx = np.sort(np.random.default_rng(seed).choice(idx, n_eval, replace=False))
    return idx, b.niche[idx], int(min(len(keep), max_k))


def nmi_ari(emb: np.ndarray, labels: np.ndarray, k: int, seed: int = 0) -> dict:
    km = MiniBatchKMeans(k, n_init=3, random_state=seed, batch_size=4096).fit(emb)
    return {"nmi": float(normalized_mutual_info_score(labels, km.labels_)), "ari": float(adjusted_rand_score(labels, km.labels_))}


def train_set(b: Brain, tissues: list[str], cap: int, seed: int, sections: list[str] | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed); out = []
    for t in tissues:
        idx = np.where(b.tissue == t)[0]
        if sections is not None:
            idx = idx[np.isin(b.section[idx], sections)]
        out.append(rng.choice(idx, min(cap, len(idx)), replace=False))
    return np.sort(np.concatenate(out))


def baselines(b: Brain, idx: np.ndarray, labels: np.ndarray, k: int, seed: int = 0) -> dict:
    """No transfer needed: PCA is fit on the evaluated tissue itself (generous to the baselines).

    nbr_mean:      mean PCA of the 15 nearest cells (one hop of hand-crafted spatial smoothing)
    nbr_mean_2hop: the same applied twice (same receptive field as the encoder, no learning)
    composition_labels: cell-class composition of the 15 nearest cells (uses the dataset's cell-class labels)
    """
    pca = GenePca(50, seed).fit(b.X, idx)
    Z = pca.transform(b.X)
    nb1 = Z[b.nbr[idx]].mean(1)
    need = np.unique(b.nbr[idx].ravel())
    n1_need = np.vstack([Z[b.nbr[need[i:i + 50_000]]].mean(1) for i in range(0, len(need), 50_000)])
    pos = np.full(len(b), -1, np.int64); pos[need] = np.arange(len(need))
    nb2 = (nb1 + n1_need[pos[b.nbr[idx]]].mean(1)) / 2
    classes, ci = np.unique(b.cell_class, return_inverse=True)
    comp = np.zeros((len(idx), len(classes)), np.float32)
    for j in range(b.nbr.shape[1]):
        np.add.at(comp, (np.arange(len(idx)), ci[b.nbr[idx, j]]), 1.0 / b.nbr.shape[1])
    rng = np.random.default_rng(seed)
    return {"expr_pca": nmi_ari(Z[idx][:, :32], labels, k, seed), "nbr_mean": nmi_ari(nb1[:, :32], labels, k, seed),
            "nbr_mean_2hop": nmi_ari(nb2[:, :32], labels, k, seed), "composition_labels": nmi_ari(comp, labels, k, seed),
            "random": nmi_ari(rng.normal(size=(len(idx), 32)), labels, k, seed)}


def run_model(b: Brain, train_nodes: np.ndarray, eval_sets: dict, steps: int, seed: int, device: str, *, shuffle: bool = False,
              trained: bool = True, b_eval: Brain | None = None, objective: str = "swav", pca_rows: np.ndarray | None = None) -> tuple[dict, list]:
    """PCA fit on the training cells, an encoder trained on them, and the NMI / ARI of each eval set (tissues of `b`, or of `b_eval`)."""
    pca = GenePca(50, seed).fit(b.X, train_nodes if pca_rows is None else pca_rows)   # pca_rows: diagnostic only (gene basis fit on other cells)
    nbr = shuffled_graph(b.nbr, b.section, seed) if shuffle else b.nbr
    enc, hist = train(pca.transform(b.X), nbr, b.section, train_nodes, steps=steps if trained else 0, seed=seed, device=device, objective=objective)
    be = b_eval or b
    Ze = pca.transform(be.X) if b_eval is not None else None
    if Ze is None:
        Ze = pca.transform(be.X)
    ne = shuffled_graph(be.nbr, be.section, seed) if shuffle else be.nbr
    out = {}
    for name, (idx, labels, k) in eval_sets.items():
        out[name] = nmi_ari(embed(enc, Ze, ne, idx, device), labels, k, seed)
    return out, hist
