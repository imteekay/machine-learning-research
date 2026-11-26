"""Features, the contrastive dual encoder (scCLIP-style) and the Gromov-Wasserstein matchers.

Everything that learns (HVGs, PCA, TF-IDF, LSI, encoders) is fit on the 3 training sites only.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA, TruncatedSVD
from torch import nn


def cp10k_log(X: sp.csr_matrix) -> sp.csr_matrix:
    lib = np.asarray(X.sum(1)).ravel()
    Y = (sp.diags(1e4 / np.maximum(lib, 1)) @ X).tocsr()
    Y.data = np.log1p(Y.data)
    return Y


class RnaFeatures:
    """CP10k-log -> top-n HVG (dispersion in mean bins, train only) -> scale, clip -> PCA."""

    def __init__(self, n_hvg: int = 3000, n_pc: int = 100, seed: int = 0):
        self.n_hvg, self.n_pc, self.seed = n_hvg, n_pc, seed

    def fit(self, counts: sp.csr_matrix):
        L = cp10k_log(counts)
        mean = np.asarray(L.mean(0)).ravel(); sq = np.asarray(L.multiply(L).mean(0)).ravel()
        var = sq - mean**2
        disp = np.log(np.maximum(var, 1e-12) / np.maximum(mean, 1e-12))
        bins = np.digitize(mean, np.quantile(mean, np.linspace(0, 1, 21)[1:-1]))
        z = np.zeros_like(disp)
        for b in np.unique(bins):
            m = bins == b; sd = disp[m].std()
            z[m] = (disp[m] - disp[m].mean()) / (sd if sd > 0 else 1)
        self.hvg = np.argsort(-z)[: self.n_hvg]
        D = L[:, self.hvg].toarray()
        self.mu, self.sd = D.mean(0), D.std(0) + 1e-6
        self.pca = PCA(self.n_pc, random_state=self.seed).fit(np.clip((D - self.mu) / self.sd, -10, 10))
        return self

    def logn(self, counts):  # normalised matrix on HVGs (for the gene-activity baseline)
        return cp10k_log(counts)[:, self.hvg].toarray()

    def transform(self, counts) -> np.ndarray:
        D = self.logn(counts)
        return self.pca.transform(np.clip((D - self.mu) / self.sd, -10, 10)).astype(np.float32)


class AtacFeatures:
    """Binarise -> TF-IDF (train document frequency) -> truncated SVD -> drop component 1 (depth)."""

    def __init__(self, n_comp: int = 100, min_cells: int = 100, seed: int = 0):
        self.n_comp, self.min_cells, self.seed = n_comp, min_cells, seed

    def _tfidf(self, P: sp.csr_matrix) -> sp.csr_matrix:
        B = P[:, self.keep].copy(); B.data[:] = 1
        tf = sp.diags(1.0 / np.maximum(np.asarray(B.sum(1)).ravel(), 1)) @ B
        X = (tf @ sp.diags(self.idf)).tocsr()
        X.data = np.log1p(X.data * 1e4)
        return X

    def fit(self, peaks: sp.csr_matrix):
        B = peaks.copy(); B.data[:] = 1
        df = np.asarray(B.sum(0)).ravel()
        self.keep = np.where(df >= self.min_cells)[0]
        self.idf = np.log1p(peaks.shape[0] / (1 + df[self.keep]))
        X = self._tfidf(peaks)
        self.svd = TruncatedSVD(self.n_comp + 1, algorithm="randomized", n_iter=5, random_state=self.seed).fit(X)
        return self

    def transform(self, peaks) -> np.ndarray:
        return self.svd.transform(self._tfidf(peaks))[:, 1:].astype(np.float32)


class DualEncoder(nn.Module):
    def __init__(self, d_rna: int, d_atac: int, d_hidden: int = 256, d_out: int = 128):
        super().__init__()
        mk = lambda d: nn.Sequential(nn.Linear(d, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(0.1),
                                     nn.Linear(d_hidden, d_out))
        self.f_rna, self.f_atac = mk(d_rna), mk(d_atac)
        self.log_t = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def encode(self, xr, xa):
        return nn.functional.normalize(self.f_rna(xr), dim=-1), nn.functional.normalize(self.f_atac(xa), dim=-1)


def train_clip(Xr: np.ndarray, Xa: np.ndarray, epochs: int = 30, batch: int = 512, lr: float = 1e-3, seed: int = 0,
               device: str = "cpu", shuffle_pairs: bool = False, trained: bool = True) -> tuple[DualEncoder, dict, list]:
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    mr, sr = Xr.mean(0), Xr.std(0) + 1e-6; ma, sa = Xa.mean(0), Xa.std(0) + 1e-6
    scale = {"mr": mr, "sr": sr, "ma": ma, "sa": sa}
    A = torch.from_numpy((Xr - mr) / sr).float().to(device)
    B = torch.from_numpy((Xa - ma) / sa).float().to(device)
    if shuffle_pairs:
        B = B[torch.from_numpy(rng.permutation(len(B))).to(device)]
    model = DualEncoder(A.shape[1], B.shape[1]).to(device)
    hist = []
    if trained:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        steps = epochs * (len(A) // batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
        model.train()
        for ep in range(epochs):
            perm = torch.from_numpy(rng.permutation(len(A))).to(device)
            tot = 0.0
            for i in range(0, len(A) - batch + 1, batch):
                j = perm[i:i + batch]
                zr, za = model.encode(A[j] + 0.05 * torch.randn_like(A[j]), B[j] + 0.05 * torch.randn_like(B[j]))
                logits = model.log_t.exp().clamp(max=100) * zr @ za.T
                y = torch.arange(len(j), device=device)
                loss = 0.5 * (nn.functional.cross_entropy(logits, y) + nn.functional.cross_entropy(logits.T, y))
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                tot += loss.item()
            hist.append(tot / max(1, len(A) // batch))
    return model.eval(), scale, hist


@torch.no_grad()
def embed(model: DualEncoder, scale: dict, Xr: np.ndarray, Xa: np.ndarray, device: str = "cpu"):
    r = torch.from_numpy((Xr - scale["mr"]) / scale["sr"]).float().to(device)
    a = torch.from_numpy((Xa - scale["ma"]) / scale["sa"]).float().to(device)
    zr, za = model.encode(r, a)
    return zr.cpu().numpy(), za.cpu().numpy()


def pdist_norm(X: np.ndarray) -> np.ndarray:
    sq = (X**2).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * X @ X.T, 0))
    return D / D.max()


def gw_plan(Er: np.ndarray, Ea: np.ndarray, M: np.ndarray | None = None, alpha: float = 0.5, max_iter: int = 200) -> np.ndarray:
    """Gromov-Wasserstein (or fused GW with a label-cost matrix M) transport plan between the two pools."""
    import ot

    n = len(Er)
    p = q = np.full(n, 1.0 / n)
    C1, C2 = pdist_norm(Er), pdist_norm(Ea)
    if M is None:
        T = ot.gromov.gromov_wasserstein(C1, C2, p, q, "square_loss", max_iter=max_iter, tol_rel=1e-7, tol_abs=1e-9)
    else:
        T = ot.gromov.fused_gromov_wasserstein(M, C1, C2, p, q, "square_loss", alpha=alpha, max_iter=max_iter, tol_rel=1e-7, tol_abs=1e-9)
    return np.asarray(T)
