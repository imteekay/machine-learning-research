"""Mini OmiCLIP: CNN image encoder + gene-sentence encoder trained with symmetric InfoNCE."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

from .simulate import GENE_IDX, GENES


class ImageEncoder(nn.Module):
    """Stand-in for OmiCLIP's ViT image tower (a pathology-FM-sized model in the paper)."""

    def __init__(self, dim: int = 64):
        super().__init__()
        c = [3, 16, 32, 64]
        layers = []
        for i in range(3):
            layers += [nn.Conv2d(c[i], c[i + 1], 3, padding=1), nn.BatchNorm2d(c[i + 1]),
                       nn.ReLU(), nn.MaxPool2d(2)]
        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(64, dim)

    def forward(self, x):
        h = self.conv(x).mean((2, 3))  # global average pool
        return self.head(h)


class SentenceEncoder(nn.Module):
    """Stand-in for OmiCLIP's text transformer: gene-token + rank-position embeddings, mean-pooled."""

    def __init__(self, k: int, dim: int = 64):
        super().__init__()
        self.tok = nn.Embedding(len(GENES), dim)
        self.pos = nn.Embedding(k, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, ids):  # ids (B, k)
        pos = torch.arange(ids.shape[1], device=ids.device)
        return self.mlp((self.tok(ids) + self.pos(pos)).mean(1))


class MiniOmiCLIP(nn.Module):
    def __init__(self, k: int, dim: int = 64):
        super().__init__()
        self.img = ImageEncoder(dim)
        self.txt = SentenceEncoder(k, dim)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def encode_image(self, x):
        return F.normalize(self.img(x), dim=-1)

    def encode_text(self, ids):
        return F.normalize(self.txt(ids), dim=-1)

    def loss(self, x, ids):
        zi, zt = self.encode_image(x), self.encode_text(ids)
        logits = self.logit_scale.exp().clamp(max=100) * zi @ zt.T
        y = torch.arange(len(x))
        return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))


def sentences_to_ids(sentences: list[list[str]]) -> np.ndarray:
    return np.array([[GENE_IDX[g] for g in s] for s in sentences], dtype=np.int64)


def _augment(x: torch.Tensor) -> torch.Tensor:
    # tissue has no canonical orientation: random rot90 + flip
    k = int(torch.randint(0, 4, ()).item())
    x = torch.rot90(x, k, (2, 3))
    if torch.rand(()) < 0.5:
        x = x.flip(3)
    return x


def train_clip(images: np.ndarray, ids: np.ndarray, epochs: int = 30, batch: int = 256,
               lr: float = 2e-3, seed: int = 0, shuffle_pairs: bool = False) -> tuple[MiniOmiCLIP, list[float]]:
    """Train mini OmiCLIP. shuffle_pairs=True is the negative control (pairs broken once, fixed)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if shuffle_pairs:
        ids = ids[rng.permutation(len(ids))]
    model = MiniOmiCLIP(ids.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    X, T = torch.from_numpy(images), torch.from_numpy(ids)
    hist = []
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        tot = 0.0
        for s in range(0, len(X) - batch + 1, batch):
            b = perm[s:s + batch]
            loss = model.loss(_augment(X[b]), T[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        hist.append(tot / max(1, len(X) // batch))
    model.eval()
    return model, hist


@torch.no_grad()
def embed(model: MiniOmiCLIP, images: np.ndarray | None = None, ids: np.ndarray | None = None):
    if images is not None:
        return torch.cat([model.encode_image(torch.from_numpy(images[i:i + 1024]))
                          for i in range(0, len(images), 1024)]).numpy()
    return model.encode_text(torch.from_numpy(ids)).numpy()


def knn_predict(query: np.ndarray, keys: np.ndarray, values: np.ndarray, k: int = 20) -> np.ndarray:
    """Loki-PredEx-style: expression of a query image = mean expression of its k most similar
    training spots in the shared space (my take: paper weights by similarity; plain mean here)."""
    sim = query @ keys.T
    top = np.argpartition(-sim, k, axis=1)[:, :k]
    return values[top].mean(1)


def handcrafted_features(images: np.ndarray) -> np.ndarray:
    """What a pathologist-inspired pipeline would measure: colour stats, nuclear area, blob count/size."""
    feats = []
    for im in images:
        rgb = im.reshape(3, -1)
        inten = rgb.mean(0)
        dark = (inten < 0.35).reshape(im.shape[1:])
        med = ((inten >= 0.35) & (inten < 0.55)).reshape(im.shape[1:])
        lab, n_dark = ndimage.label(dark)
        _, n_med = ndimage.label(med)
        size_dark = dark.sum() / max(n_dark, 1)
        feats.append(np.r_[rgb.mean(1), rgb.std(1), dark.mean(), med.mean(), n_dark, n_med, size_dark])
    return np.asarray(feats, dtype=np.float64)


def ridge_fit_predict(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, lam: float = 1.0) -> np.ndarray:
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = Ytr.mean(0)
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (Ytr - ym))
    return B @ W + ym
