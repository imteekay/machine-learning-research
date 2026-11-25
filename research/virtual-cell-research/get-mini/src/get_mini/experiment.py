"""E3.1: is the ATAC information regulatory?

Per seed:
1. Mask 20% of genes (their RNA is hidden) and hold out 30% of metacells.
2. For each masked gene, fit on train metacells and score Pearson r on test metacells:
   - rna:          ridge on a PCA of the *visible* genes' RNA (cell state from other genes)
   - rna_atac:     + this gene's linked peaks (region accessibility)
   - rna_shuf:     + the linked peaks of a *random other gene* (negative control: same #features,
                   same generic cell-type info in ATAC, wrong regulatory elements)
   - rna_gas:      + one gene-activity score (distance-weighted sum of nearby accessibility)
   - gas:          zero-parameter gene-activity score alone (classic ATAC->RNA baseline)
   - atac, atac_shuf: the gene's own vs a random gene's peaks, without RNA
3. gain = r(rna_atac) - r(rna); test Spearman(gain, tau) and Spearman(gain, #peaks),
   and Wilcoxon gain vs gain_shuf.
4. TinyGET trained on visible genes x train metacells, evaluated on masked genes x test metacells,
   vs the same model on gene-shuffled regions.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .models import pearson, predict_tinyget, ridge_gcv_fit_predict, train_tinyget
from .preprocess import Pseudobulk, pca, tau_specificity

N_TOK_FEAT = 5


def region_tokens(atac_z: np.ndarray, atac_log: np.ndarray, links, genes, K: int):
    """Build (len(genes), M, K, F) tokens and (len(genes), K) pad mask."""
    M = atac_z.shape[0]
    tok = np.zeros((len(genes), M, K, N_TOK_FEAT), dtype=np.float32)
    pad = np.ones((len(genes), K), dtype=bool)
    for i, g in enumerate(genes):
        idx, d, inside = links[g]
        k = min(len(idx), K)
        if k == 0:
            continue
        ld = np.log10(np.abs(d[:k]) + 1) / 5
        tok[i, :, :k, 0] = atac_z[:, idx[:k]]
        tok[i, :, :k, 1] = atac_log[:, idx[:k]]
        tok[i, :, :k, 2] = (np.sign(d[:k]) * ld)[None]
        tok[i, :, :k, 3] = ld[None]
        tok[i, :, :k, 4] = inside[:k][None]
        pad[i, :k] = False
    return tok, pad


def gene_activity(atac_log: np.ndarray, links, g: int) -> np.ndarray:
    idx, d, _ = links[g]
    w = np.exp(-np.abs(d) / 20_000)  # promoter/gene-body peaks weigh most, distal ones decay
    return np.log1p((np.expm1(atac_log[:, idx]) * w).sum(1))


def run_seed(pb: Pseudobulk, links, genes: np.ndarray, seed: int, n_pcs: int = 10, K: int = 32,
             mask_frac: float = 0.2, test_frac: float = 0.3, tinyget: bool = True, epochs: int = 8):
    rng = np.random.default_rng(seed)
    M = pb.rna.shape[0]
    perm = rng.permutation(M)
    te = np.sort(perm[: int(round(test_frac * M))])
    tr = np.sort(perm[int(round(test_frac * M)):])
    gperm = rng.permutation(genes)
    n_mask = int(round(mask_frac * len(genes)))
    masked, visible = np.sort(gperm[:n_mask]), np.sort(gperm[n_mask:])

    Y = pb.rna  # metacells x genes (log CPM)
    Zrna, _ = pca(Y[:, visible], n_pcs, fit_rows=tr)
    A = pb.atac
    a_mu, a_sd = A[tr].mean(0), A[tr].std(0)
    a_sd[a_sd == 0] = 1
    Az = (A - a_mu) / a_sd

    # tau over cell clusters, from train metacells, on linear scale
    clusters = np.unique(pb.cluster[tr])
    means = np.stack([np.expm1(Y[tr][pb.cluster[tr] == c]).mean(0) for c in clusters])
    tau = tau_specificity(means)

    shuf_src = rng.permutation(genes)  # gene -> random other gene's links
    shuf_map = dict(zip(genes, shuf_src))

    rows = []
    for g in masked:
        y = Y[:, g]
        idx = links[g][0][:K]
        sidx = links[shuf_map[g]][0][:K]
        gas = gene_activity(A, links, g)
        feats = {
            "rna": Zrna,
            "rna_atac": np.hstack([Zrna, Az[:, idx]]),
            "rna_shuf": np.hstack([Zrna, Az[:, sidx]]),
            "rna_gas": np.hstack([Zrna, gas[:, None]]),
            "atac": Az[:, idx],
            "atac_shuf": Az[:, sidx],
        }
        r = {k: pearson(ridge_gcv_fit_predict(X[tr], y[tr], X[te]), y[te]) for k, X in feats.items()}
        r["gas"] = pearson(gas[te], y[te])
        rows.append(dict(gene=int(g), tau=float(tau[g]), n_peaks=int(len(links[g][0])),
                         mean_expr=float(Y[:, g].mean()), **{f"r_{k}": v for k, v in r.items()}))

    out = {"rows": rows, "n_train_mc": int(len(tr)), "n_test_mc": int(len(te)), "n_masked": int(n_mask)}

    if tinyget:
        Ymu, Ysd = Y[tr].mean(0), Y[tr].std(0)
        Ysd[Ysd == 0] = 1
        Yz = (Y - Ymu) / Ysd  # (train-metacell stats, applied to all)
        res = {}
        for name, lk in [("real", links), ("shuf", {g: links[shuf_map[g]] for g in genes})]:
            tok_v, pad_v = region_tokens(Az, A, lk, visible, K)
            Xtr = tok_v[:, tr].reshape(-1, K, N_TOK_FEAT)
            Ptr = np.repeat(pad_v, len(tr), 0)
            ytr = Yz[tr][:, visible].T.reshape(-1)
            model = train_tinyget(Xtr, Ptr, ytr, epochs=epochs, seed=seed)
            tok_m, pad_m = region_tokens(Az, A, lk, masked, K)
            Xte = tok_m[:, te].reshape(-1, K, N_TOK_FEAT)
            pred, att = predict_tinyget(model, Xte, np.repeat(pad_m, len(te), 0))
            pred = pred.reshape(len(masked), len(te))
            res[name] = [pearson(pred[i], Yz[te, g]) for i, g in enumerate(masked)]
            if name == "real":
                # mean attention by distance bin (what the pooled query looks at)
                att = att.reshape(len(masked), len(te), K).mean(1)
                dist = np.full((len(masked), K), np.nan)
                for i, g in enumerate(masked):
                    d = np.abs(links[g][1][:K])
                    dist[i, : len(d)] = d
                out["attention_by_distance"] = _attn_by_distance(att, dist, pad_m)
        for i, row in enumerate(rows):
            row["r_tinyget"] = float(res["real"][i])
            row["r_tinyget_shuf"] = float(res["shuf"][i])
    return out


def _attn_by_distance(att, dist, pad):
    bins = [0, 1, 5_000, 20_000, 50_000, 100_001]
    labels = ["in-body", "<5kb", "5-20kb", "20-50kb", "50-100kb"]
    n_real = (~pad).sum(1, keepdims=True)
    rel = att * n_real  # >1 means above-uniform attention
    out = {}
    for lab, lo, hi in zip(labels, bins[:-1], bins[1:]):
        m = (~pad) & (dist >= lo) & (dist < hi)
        out[lab] = float(rel[m].mean()) if m.any() else None
    return out


def summarize(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if k.startswith("r_")]
    arr = {k: np.array([r[k] for r in rows]) for k in keys}
    tau = np.array([r["tau"] for r in rows])
    npk = np.array([r["n_peaks"] for r in rows])
    gain = arr["r_rna_atac"] - arr["r_rna"]
    gain_shuf = arr["r_rna_shuf"] - arr["r_rna"]
    s = {"median_r": {k: float(np.median(v)) for k, v in arr.items()}}
    s["median_gain"] = float(np.median(gain))
    s["median_gain_shuf"] = float(np.median(gain_shuf))
    s["median_gain_gas"] = float(np.median(arr["r_rna_gas"] - arr["r_rna"]))
    s["wilcoxon_gain_vs_shuf_p"] = float(stats.wilcoxon(gain, gain_shuf, alternative="greater").pvalue)
    rho, p = stats.spearmanr(gain, tau)
    s["spearman_gain_tau"] = {"rho": float(rho), "p": float(p)}
    rho, p = stats.spearmanr(gain - gain_shuf, tau)
    s["spearman_specificgain_tau"] = {"rho": float(rho), "p": float(p)}
    rho, p = stats.spearmanr(gain_shuf, tau)
    s["spearman_gainshuf_tau"] = {"rho": float(rho), "p": float(p)}
    rho, p = stats.spearmanr(gain, npk)
    s["spearman_gain_npeaks"] = {"rho": float(rho), "p": float(p)}
    q = np.quantile(tau, [0, 0.25, 0.5, 0.75, 1.0])
    s["gain_by_tau_quartile"] = []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (tau >= lo) & (tau <= hi)
        s["gain_by_tau_quartile"].append({"tau_lo": float(lo), "tau_hi": float(hi), "n": int(m.sum()),
                                          "median_gain": float(np.median(gain[m])),
                                          "median_gain_shuf": float(np.median(gain_shuf[m]))})
    nb = [(1, 5), (6, 15), (16, 10_000)]
    s["gain_by_npeaks"] = [{"bin": f"{lo}-{hi if hi < 10_000 else '+'}", "n": int(((npk >= lo) & (npk <= hi)).sum()),
                            "median_gain": float(np.median(gain[(npk >= lo) & (npk <= hi)]))
                            if ((npk >= lo) & (npk <= hi)).any() else None} for lo, hi in nb]
    if "r_tinyget" in arr:
        s["wilcoxon_tinyget_vs_shuf_p"] = float(stats.wilcoxon(arr["r_tinyget"], arr["r_tinyget_shuf"],
                                                                 alternative="greater").pvalue)
        s["wilcoxon_tinyget_vs_gas_p"] = float(stats.wilcoxon(arr["r_tinyget"], arr["r_gas"],
                                                                alternative="greater").pvalue)
    return s
