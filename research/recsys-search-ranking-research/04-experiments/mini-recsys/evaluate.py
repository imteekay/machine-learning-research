"""Offline metrics: retrieval (full catalogue), in-search ranking, embedding health, policy."""
from __future__ import annotations

import numpy as np


def retrieval_metrics(scores: np.ndarray, positives: list[np.ndarray], k=10) -> dict:
    """scores (Q, N) over the full catalogue; positives[q] = relevant house ids."""
    top = np.argsort(-scores, 1)[:, :k]
    hit, rec, ndcg, mrr = [], [], [], []
    order = np.argsort(-scores, 1)
    for q, pos in enumerate(positives):
        pos = set(pos.tolist())
        rel = np.array([t in pos for t in top[q]])
        hit.append(rel.any())
        rec.append(rel.sum() / len(pos))
        dcg = (rel / np.log2(np.arange(2, k + 2))).sum()
        idcg = (1 / np.log2(np.arange(2, min(len(pos), k) + 2))).sum()
        ndcg.append(dcg / idcg)
        first = np.flatnonzero(np.isin(order[q], list(pos)))[0]
        mrr.append(1 / (first + 1))
    return {f"hit@{k}": float(np.mean(hit)), f"recall@{k}": float(np.mean(rec)),
            f"ndcg@{k}": float(np.mean(ndcg)), "mrr": float(np.mean(mrr)),
            f"coverage@{k}": len(np.unique(top)) / scores.shape[1]}


def session_ndcg(scores: np.ndarray, labels: np.ndarray, k=5) -> float:
    """Mean NDCG@k over sessions (rows) with at least one positive. Ties broken randomly."""
    keep = labels.sum(1) > 0
    s, y = scores[keep], labels[keep]
    jitter = np.random.default_rng(0).random(s.shape) * 1e-9
    order = np.argsort(-(s + jitter), 1)[:, :k]
    gains = np.take_along_axis(y, order, 1)
    disc = 1 / np.log2(np.arange(2, k + 2))
    dcg = (gains * disc).sum(1)
    ideal = -np.sort(-y, 1)[:, :k]
    return float((dcg / (ideal * disc).sum(1)).mean())


def information_abundance(emb: np.ndarray) -> float:
    """sum(singular values) / max(singular value): 1 = collapsed, d = all dims used."""
    sv = np.linalg.svd(emb - emb.mean(0), compute_uv=False)
    return float(sv.sum() / sv.max())


def policy_tradeoff(scores, labels, demand, hot, gammas, k=5, top=3):
    """score * (1 - demand)^gamma, demand in [0, 1). Concentration = share of the first
    `top` slots held by "hot" houses (top decile of current demand)."""
    rows = []
    for g in gammas:
        s = scores * (1 - demand) ** g
        order = np.argsort(-s, 1)[:, :top]
        conc = float(np.take_along_axis(hot, order, 1).mean())
        rows.append({"gamma": g, "ndcg@5": session_ndcg(s, labels, k), "hot_share@3": conc})
    return rows
