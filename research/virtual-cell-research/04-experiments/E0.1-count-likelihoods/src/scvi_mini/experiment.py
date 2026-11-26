"""Train every likelihood on the same 80/20 cell split; score held-out cells in count space."""

from __future__ import annotations

import numpy as np
import torch

from .data import Counts
from .models import MiniSCVI, fit, to_tensor

MODELS = {  # name -> (kind, n_latent, shuffle_genes)
    "poisson": ("poisson", 10, False),
    "nb": ("nb", 10, False),
    "zinb": ("zinb", 10, False),
    "gaussian_log1p": ("gaussian", 10, False),
    "nb_floor": ("nb", 0, False),          # no latent: per-gene NB only
    "nb_shuffled": ("nb", 10, True),       # genes permuted across cells in training: no covariance to learn
}


def split(n: int, seed: int, frac: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    k = int(frac * n)
    return np.sort(perm[:k]), np.sort(perm[k:])


def shuffle_genes(X: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    out = X.clone()
    for j in range(X.shape[1]):
        out[:, j] = X[torch.randperm(X.shape[0], generator=g), j]
    return out


@torch.no_grad()
def heldout_ll(model: MiniSCVI, X, lib, lib_total, batch: int = 1024) -> np.ndarray:
    """Mean log P(count) per held-out cell (averaged over genes), z = posterior mean."""
    model.eval()
    out = []
    for i in range(0, X.shape[0], batch):
        d, _ = model.params(X[i:i + batch], lib[i:i + batch], sample=False)
        out.append(model.count_logpmf(X[i:i + batch], d, lib_total[i:i + batch]).mean(-1))
    return torch.cat(out).numpy()


@torch.no_grad()
def ppc(model: MiniSCVI, X, lib, lib_total, seed: int) -> dict:
    """Posterior-predictive checks on per-gene summaries of the held-out cells."""
    gen = torch.Generator().manual_seed(seed)
    Xs = model.sample_counts(X, lib, lib_total, gen)
    obs, sim = X.numpy(), Xs.numpy()
    m_o, v_o, z_o = obs.mean(0), obs.var(0), (obs == 0).mean(0)
    m_s, v_s, z_s = sim.mean(0), sim.var(0), (sim == 0).mean(0)
    ok = (m_o > 0) & (v_o > 0) & (v_s > 0)
    return {
        "median_abs_log_var_ratio": float(np.median(np.abs(np.log(v_s[ok] / v_o[ok])))),
        "median_abs_log_mean_ratio": float(np.median(np.abs(np.log((m_s[ok] + 1e-8) / m_o[ok])))),
        "zero_frac_mae": float(np.mean(np.abs(z_s - z_o))),
        "curve": {"mean_obs": m_o.tolist(), "var_obs": v_o.tolist(), "var_sim": v_s.tolist(),
                  "zero_obs": z_o.tolist(), "zero_sim": z_s.tolist()},
    }


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean of (a - b) over cells with a 95% percentile CI."""
    d = a - b
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci95": [float(lo), float(hi)]}


def run(ds: Counts, seed: int, epochs: int, models: dict = MODELS) -> dict:
    tr, te = split(ds.X.shape[0], seed)
    X = to_tensor(ds.X)
    lib = torch.from_numpy(ds.lib.astype(np.float32))
    libt = torch.from_numpy(ds.lib_total.astype(np.float32))
    res: dict = {"n_train": int(len(tr)), "n_test": int(len(te)), "models": {}}
    lls = {}
    for name, (kind, nl, shuf) in models.items():
        Xtr = shuffle_genes(X[tr], seed) if shuf else X[tr]
        libtr = Xtr.sum(1)
        m = MiniSCVI(X.shape[1], kind, n_latent=nl)
        hist = fit(m, Xtr, libtr, libt[tr], epochs=epochs, seed=seed)
        ll = heldout_ll(m, X[te], lib[te], libt[te])
        lls[name] = ll
        entry = {"heldout_ll_per_entry": float(ll.mean()), "final_train_loss": hist[-1]}
        entry["ppc"] = ppc(m, X[te], lib[te], libt[te], seed)
        if kind in ("nb", "zinb") and nl > 0:
            entry["median_theta"] = float(torch.exp(m.log_theta.detach()).median())
        res["models"][name] = entry
    comps = [("nb", "poisson"), ("nb", "gaussian_log1p"), ("zinb", "nb"), ("nb", "nb_floor"), ("nb", "nb_shuffled")]
    res["paired"] = {f"{a}-{b}": paired_bootstrap(lls[a], lls[b], seed=seed) for a, b in comps if a in lls and b in lls}
    # practical-equivalence margin for ZINB vs NB: 0.5% of |NB held-out LL|
    if "nb" in lls:
        res["zinb_equiv_margin"] = float(0.005 * abs(lls["nb"].mean()))
    return res
