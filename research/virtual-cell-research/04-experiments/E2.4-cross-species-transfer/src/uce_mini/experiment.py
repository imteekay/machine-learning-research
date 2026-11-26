"""Train on mouse, probe on mouse, test on human. Every arm sees the same cells and the same labels.

Arms
- ortholog_pca: 1:1-ortholog genes, HVG, log CP10k, PCA fit on mouse, logistic regression on mouse (pre-registered baseline)
- ortholog_pca_centered: same, but each species' genes are centred on their own mean (cheap domain alignment)
- protein_uce: mini UCE with frozen ESM2 protein tokens, self-supervised on mouse cells only
- protein_untrained: same architecture and tokens, random weights (is the embedding just random features?)
- token_ortholog_random: random tokens, shared between 1:1 orthologs (token identity only, no protein information)
- token_independent_random: random tokens independent in each species (no way to transfer: sanity floor)
- shuffled_labels: logistic regression trained on shuffled mouse labels on the protein_uce embedding
- protein_nonortholog / token_independent_nonortholog: models that never see 1:1 orthologs (expression masked)
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data import Species
from .models import UCEMini, embed_cells, train


def macro_f1(y: np.ndarray, p: np.ndarray, k: int) -> float:
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    return float(f1[cm.sum(1) > 0].mean())


def shared_labels(m: Species, h: Species, min_cells: int = 50) -> list[str]:
    cm = dict(zip(*np.unique(m.cell_type, return_counts=True)))
    ch = dict(zip(*np.unique(h.cell_type, return_counts=True)))
    return sorted(k for k in cm if k in ch and cm[k] >= min_cells and ch[k] >= min_cells)


def restrict(s: Species, labels: list[str]) -> Species:
    keep = np.isin(s.cell_type, labels)
    return Species(s.X[keep], s.genes, s.cell_type[keep], s.tissue[keep], s.name)


def ortholog_index(m: Species, h: Species, pairs) -> tuple[np.ndarray, np.ndarray]:
    mi = {g: i for i, g in enumerate(m.genes)}
    hi = {g: i for i, g in enumerate(h.genes)}
    P = [(mi[b], hi[a]) for a, b in zip(pairs.human, pairs.mouse) if a in hi and b in mi]
    P = np.array(P)
    return P[:, 0], P[:, 1]


def build_tables(m: Species, h: Species, emb_m: dict, emb_h: dict, im: np.ndarray, ih: np.ndarray, seed: int) -> dict:
    Em = np.stack([emb_m[g] for g in m.genes]).astype(np.float32)
    Eh = np.stack([emb_h[g] for g in h.genes]).astype(np.float32)
    both = np.vstack([Em, Eh])
    mu, sd = both.mean(0), both.std(0) + 1e-6
    prot = {"mouse": torch.from_numpy((Em - mu) / sd), "human": torch.from_numpy((Eh - mu) / sd)}
    rng = np.random.default_rng(seed + 12345)
    d = Em.shape[1]
    rm = rng.normal(size=(len(m.genes), d)).astype(np.float32)
    rh = rng.normal(size=(len(h.genes), d)).astype(np.float32)
    rh_shared = rh.copy()
    rh_shared[ih] = rm[im]
    rind = rng.normal(size=(len(h.genes), d)).astype(np.float32)
    return {"protein": prot,
            "random_shared": {"mouse": torch.from_numpy(rm), "human": torch.from_numpy(rh_shared)},
            "random_independent": {"mouse": torch.from_numpy(rm), "human": torch.from_numpy(rind)}}


def _cp10k_log(X: sp.csr_matrix) -> sp.csr_matrix:
    lib = np.asarray(X.sum(1)).ravel()
    Y = sp.diags(1e4 / np.maximum(lib, 1)) @ X
    Y = Y.tocsr(); Y.data = np.log1p(Y.data)
    return Y


def probe(Ztr, ytr, Zte_list, seed: int, C: float = 1.0):
    sc = StandardScaler().fit(Ztr)
    clf = LogisticRegression(max_iter=1000, C=C, random_state=seed).fit(sc.transform(Ztr), ytr)
    return [clf.predict(sc.transform(Z)) for Z in Zte_list]


def pca_arm(m: Species, h: Species, im, ih, tr, te, ytr, centered: bool, seed: int, n_hvg=2000, n_pc=50):
    Xm = _cp10k_log(m.X[:, im]); Xh = _cp10k_log(h.X[:, ih])
    mean = np.asarray(Xm[tr].mean(0)).ravel()
    sq = np.asarray(Xm[tr].multiply(Xm[tr]).mean(0)).ravel()
    hvg = np.argsort(-(sq - mean**2))[:n_hvg]
    A, B = Xm[:, hvg].toarray(), Xh[:, hvg].toarray()
    mu_m = A[tr].mean(0)
    mu_h = B.mean(0) if centered else mu_m
    pca = PCA(n_pc, random_state=seed).fit(A[tr] - mu_m)
    Z = pca.transform(A - mu_m); Zh = pca.transform(B - mu_h)
    pm, ph = probe(Z[tr], ytr, [Z[te], Zh], seed)
    return pm, ph


def uce_arm(m, h, tabs, tr, te, ytr, steps, seed, device, trained=True, mask_cols=None, mask_cols_h=None, L=128):
    Xm, Xh = m.X, h.X
    if mask_cols is not None:  # hide the 1:1 orthologs: the model only ever sees non-orthologous genes
        keep_m = sp.diags((~mask_cols).astype(np.float32)); keep_h = sp.diags((~mask_cols_h).astype(np.float32))
        Xm, Xh = (Xm @ keep_m).tocsr(), (Xh @ keep_h).tocsr()
    torch.manual_seed(seed)
    model = UCEMini(d_tok=tabs["mouse"].shape[1])
    hist = train(model, Xm[tr], tabs["mouse"], steps, seed=seed, device=device, L=L) if trained else []
    Z = embed_cells(model, Xm, tabs["mouse"], seed=seed, device=device, L=L)
    Zh = embed_cells(model, Xh, tabs["human"], seed=seed, device=device, L=L)
    return Z, Zh, hist


def run(m: Species, h: Species, pairs, emb_m: dict, emb_h: dict, seed: int, steps: int, device: str = "cpu",
        arms: tuple[str, ...] | None = None) -> dict:
    labels = shared_labels(m, h)
    m, h = restrict(m, labels), restrict(h, labels)
    lab = {l: i for i, l in enumerate(labels)}
    ym = np.array([lab[x] for x in m.cell_type]); yh = np.array([lab[x] for x in h.cell_type])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ym)); ntr = int(0.8 * len(ym))
    tr, te = np.sort(perm[:ntr]), np.sort(perm[ntr:])
    im, ih = ortholog_index(m, h, pairs)
    tabs = build_tables(m, h, emb_m, emb_h, im, ih, seed)
    k = len(labels)
    is_orth_m = np.zeros(len(m.genes), bool); is_orth_m[im] = True
    is_orth_h = np.zeros(len(h.genes), bool); is_orth_h[ih] = True
    preds: dict[str, np.ndarray] = {}
    sanity: dict[str, float] = {}
    hists: dict[str, list] = {}
    want = lambda a: arms is None or a in arms

    for name, cen in (("ortholog_pca", False), ("ortholog_pca_centered", True)):
        if want(name):
            pm, ph = pca_arm(m, h, im, ih, tr, te, ym[tr], cen, seed)
            preds[name] = ph; sanity[name] = macro_f1(ym[te], pm, k)
    specs = {
        "protein_uce": ("protein", True, False),
        "protein_untrained": ("protein", False, False),
        "token_ortholog_random": ("random_shared", True, False),
        "token_independent_random": ("random_independent", True, False),
        "protein_nonortholog": ("protein", True, True),
        "token_independent_nonortholog": ("random_independent", True, True),
    }
    emb_store = {}
    for name, (tk, trained, nonorth) in specs.items():
        if not want(name):
            continue
        t = {"mouse": tabs[tk]["mouse"], "human": tabs[tk]["human"]}
        Z, Zh, hist = uce_arm(m, h, t, tr, te, ym[tr], steps, seed, device, trained,
                              is_orth_m if nonorth else None, is_orth_h if nonorth else None)
        pm, ph = probe(Z[tr], ym[tr], [Z[te], Zh], seed)
        preds[name] = ph; sanity[name] = macro_f1(ym[te], pm, k); hists[name] = hist[::max(1, len(hist) // 50)]
        emb_store[name] = (Z, Zh)
    if "protein_uce" in emb_store:
        Z, Zh = emb_store["protein_uce"]
        (ph,) = probe(Z[tr], rng.permutation(ym[tr]), [Zh], seed)
        preds["shuffled_labels"] = ph
    maj = np.bincount(ym[tr], minlength=k).argmax()
    preds["majority_class"] = np.full(len(yh), maj)
    f1 = {a: macro_f1(yh, p, k) for a, p in preds.items()}
    per_label = {a: {labels[i]: float(((p == i) & (yh == i)).sum() * 2 / max((p == i).sum() + (yh == i).sum(), 1))
                     for i in range(k)} for a, p in preds.items()}
    return {"seed": seed, "labels": labels, "n_mouse": int(len(ym)), "n_human": int(len(yh)),
            "n_orthologs": int(len(im)), "frac_nonortholog_genes": {"mouse": float(1 - is_orth_m.mean()), "human": float(1 - is_orth_h.mean())},
            "human_macro_f1": f1, "mouse_heldout_macro_f1": sanity, "per_label_f1": per_label,
            "train_loss": hists, "_preds": preds, "_yh": yh, "_k": k}


def paired_bootstrap(yh: np.ndarray, k: int, preds_by_seed: list[dict], a: str, b: str, n_boot: int = 500, seed: int = 0) -> dict:
    """Difference in seed-averaged macro-F1 between arms a and b, resampling human cells."""
    rng = np.random.default_rng(seed)
    n = len(yh)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        fa = np.mean([macro_f1(yh[idx], p[a][idx], k) for p in preds_by_seed])
        fb = np.mean([macro_f1(yh[idx], p[b][idx], k) for p in preds_by_seed])
        diffs.append(fa - fb)
    full = np.mean([macro_f1(yh, p[a], k) - macro_f1(yh, p[b], k) for p in preds_by_seed])
    return {"mean": float(full), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
