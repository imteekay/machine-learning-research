"""The toy cell: a small "wet lab" whose causal truth is known.

A real virtual cell learns from measurements (Perturb-seq) whose causal truth nobody knows. Here we
build the world ourselves, so every prediction can be checked against the truth:

    gene regulatory network (GRN) ──► latent log-expression x ──► noisy UMI counts
                 ▲                                ▲
       knockdown of gene k              stress program shared by every perturbed cell

**Latent model.** Genes are in topological order, so regulators come before their children:

    x_i = b_i + Σ_j w_ji · tanh(x_j − b_j) + σ · ε_i

``b_i`` is exactly the control mean of ``x_i``, because tanh is odd and ε is symmetric. A regulator
pushes its children only when it deviates from its own baseline:

- Small natural fluctuations sit in the near-linear part of tanh. That is what makes a regulator
  co-expressed with its targets in control cells.
- A knockdown, several log-units deep, saturates tanh.

**Perturb-seq in silico.**

- *CRISPRi knockdown of gene k*: in responder cells ``x_k`` drops by ``kd`` log-units, and the drop
  propagates to every descendant. Escaper cells keep ``x_k`` (incomplete knockdown).
- *Systematic variation*: every cell that received a targeting guide gets ``γ_k · s``, a shared
  "stress program" on a few non-regulator genes. This is the shortcut Systema warns about: a model
  can score well by predicting only this generic response.
- *Counts*: ``μ = s_c · exp(x)``, and counts ~ NegativeBinomial(μ, θ), drawn as Gamma-Poisson.
  ``s_c`` is a per-cell capture/library factor, which normalization removes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data import CONTROL, normalize_log1p


@dataclass(frozen=True)
class ToyCellConfig:
    """Every knob of the toy world. The defaults are the frozen E0.4 world."""

    # gene regulatory network
    n_genes: int = 500
    n_regulators: int = 80
    n_masters: int = 10  # regulators without parents
    max_regulator_parents: int = 2
    max_target_parents: int = 3
    weight_range: tuple[float, float] = (0.3, 1.0)  # |w|
    p_activating: float = 0.7
    # expression and measurement
    baseline_mean: float = 1.0
    baseline_sd: float = 0.8
    noise_sd: float = 0.3  # intrinsic (biological) noise in log space
    library_sd: float = 0.3  # capture / sequencing-depth factor
    nb_theta: float = 10.0  # negative-binomial dispersion (larger = closer to Poisson)
    # perturbations
    responder_frac: float = 0.8
    kd_range: tuple[float, float] = (1.5, 2.5)  # log-units removed in responder cells
    n_stress_genes: int = 25
    stress_range: tuple[float, float] = (0.3, 0.8)
    systematic_strength: float = 1.0  # 0 switches the generic stress response off
    gamma_range: tuple[float, float] = (0.5, 1.5)  # per-knockout strength of the stress response
    # screen
    n_null_perts: int = 40  # knockouts of non-regulator genes (no downstream effect)
    n_ctrl: int = 2000
    cells_per_pert: int = 200  # raised from 100 by the E0.4 pilot rule (oracle ceiling ≥ 0.7)
    oracle_cells: int = 4000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Screen:
    """A simulated Perturb-seq screen, shaped like AnnData: counts (cells × genes) + per-cell obs."""

    counts: np.ndarray  # (n_cells, n_genes) integer UMI counts
    obs: pd.DataFrame  # perturbation (gene name or "control"), target (gene index or -1), responder
    gene_names: list[str]


class ToyCell:
    """A random GRN cell. ``seed`` fixes the world: network, baselines, stress program, knockouts."""

    def __init__(self, config: ToyCellConfig | None = None, seed: Any = 0):
        cfg = config or ToyCellConfig()
        if not 0 < cfg.n_masters <= cfg.n_regulators < cfg.n_genes:
            raise ValueError("need 0 < n_masters <= n_regulators < n_genes")
        self.config = cfg
        rng = np.random.default_rng(seed)
        G, R = cfg.n_genes, cfg.n_regulators

        self.gene_names = [f"TF{i:02d}" for i in range(R)] + [f"G{i:03d}" for i in range(R, G)]
        self.is_regulator = np.arange(G) < R
        self.baseline = rng.normal(cfg.baseline_mean, cfg.baseline_sd, size=G)

        # W[j, i] is the effect of parent j on child i. Parents always have a lower index (a DAG).
        W = np.zeros((G, G))
        for i in range(cfg.n_masters, R):
            n_parents = int(rng.integers(1, cfg.max_regulator_parents + 1))
            parents = rng.choice(i, size=min(n_parents, i), replace=False)
            W[parents, i] = self._edge_weights(rng, len(parents))
        for i in range(R, G):
            n_parents = int(rng.integers(1, cfg.max_target_parents + 1))
            parents = rng.choice(R, size=min(n_parents, R), replace=False)
            W[parents, i] = self._edge_weights(rng, len(parents))
        for j in range(R):  # every regulator regulates at least one gene
            if not W[j].any():
                W[j, int(rng.integers(R, G))] = self._edge_weights(rng, 1)[0]
        self.W = W
        self.parents = [np.flatnonzero(W[:, i]) for i in range(G)]

        stress_genes = rng.choice(np.arange(R, G), size=cfg.n_stress_genes, replace=False)
        self.stress_program = np.zeros(G)
        self.stress_program[stress_genes] = rng.uniform(
            *cfg.stress_range, size=cfg.n_stress_genes
        ) * rng.choice([-1.0, 1.0], size=cfg.n_stress_genes)

        null_targets = np.sort(rng.choice(np.arange(R, G), size=cfg.n_null_perts, replace=False))
        self.perturbation_targets = np.concatenate([np.arange(R), null_targets])
        self.perturbation_names = [self.gene_names[k] for k in self.perturbation_targets]
        self.gamma = np.zeros(G)  # stress-response strength of each knockout (by target gene)
        self.gamma[self.perturbation_targets] = cfg.systematic_strength * rng.uniform(
            *cfg.gamma_range, size=len(self.perturbation_targets)
        )

    def _edge_weights(self, rng: np.random.Generator, n: int) -> np.ndarray:
        lo, hi = self.config.weight_range
        sign = np.where(rng.random(n) < self.config.p_activating, 1.0, -1.0)
        return sign * rng.uniform(lo, hi, size=n)

    # ------------------------------------------------------------------ ground truth
    @property
    def n_genes(self) -> int:
        return self.config.n_genes

    @property
    def perturbation_strata(self) -> dict[str, str]:
        """Knockout name → "regulator" (has downstream targets) or "null" (does not)."""
        return {
            self.gene_names[k]: "regulator" if self.is_regulator[k] else "null"
            for k in self.perturbation_targets
        }

    def children(self, k: int) -> np.ndarray:
        return np.flatnonzero(self.W[k])

    def descendants(self, k: int) -> np.ndarray:
        """Every gene reachable from gene ``k``: the true direct and indirect targets of a knockout."""
        seen: set[int] = set()
        stack = list(self.children(k))
        while stack:
            gene = int(stack.pop())
            if gene not in seen:
                seen.add(gene)
                stack.extend(self.children(gene))
        return np.array(sorted(seen), dtype=int)

    # ------------------------------------------------------------------ cells
    def latent(
        self, eps: np.ndarray, target: np.ndarray | None = None, kd: np.ndarray | None = None
    ) -> np.ndarray:
        """Latent log-expression of ``len(eps)`` cells, computed gene by gene in topological order.

        eps:    (n, G) standard-normal intrinsic noise.
        target: (n,) gene knocked down in each cell (-1 = none).
        kd:     (n,) log-units removed from the target (0 for escapers and controls).
        """
        x = np.empty_like(eps, dtype=float)
        b, sd = self.baseline, self.config.noise_sd
        for i in range(eps.shape[1]):
            xi = b[i] + sd * eps[:, i]
            p = self.parents[i]
            if p.size:
                xi = xi + np.tanh(x[:, p] - b[p]) @ self.W[p, i]
            if target is not None:
                hit = target == i
                if hit.any():
                    xi = xi - np.where(hit, kd, 0.0)
            x[:, i] = xi
        return x

    def add_systematic(self, x: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Add the shared stress program to every cell that received a targeting guide."""
        strength = np.where(target >= 0, self.gamma[np.maximum(target, 0)], 0.0)
        return x + strength[:, None] * self.stress_program[None, :]

    def sample_counts(
        self, x: np.ndarray, rng: np.random.Generator, library: np.ndarray | None = None
    ) -> np.ndarray:
        """UMI counts ~ NB(mean = library · exp(x), dispersion θ), drawn as a Gamma-Poisson mixture."""
        if library is None:
            library = rng.lognormal(0.0, self.config.library_sd, size=x.shape[0])
        mu = library[:, None] * np.exp(x)
        theta = self.config.nb_theta
        return rng.poisson(rng.gamma(shape=theta, scale=mu / theta))

    def simulate_screen(
        self, seed: Any = None, cells_per_pert: int | None = None, n_ctrl: int | None = None
    ) -> Screen:
        """Run the Perturb-seq screen: control cells plus ``cells_per_pert`` cells per knockout."""
        cfg = self.config
        rng = np.random.default_rng(seed)
        n_per = cells_per_pert or cfg.cells_per_pert
        n_ctrl = n_ctrl or cfg.n_ctrl
        target = np.concatenate([np.full(n_ctrl, -1), np.repeat(self.perturbation_targets, n_per)])
        n = len(target)
        responder = (target >= 0) & (rng.random(n) < cfg.responder_frac)
        kd = np.where(responder, rng.uniform(*cfg.kd_range, size=n), 0.0)
        x = self.latent(rng.standard_normal((n, cfg.n_genes)), target, kd)
        x = self.add_systematic(x, target)
        counts = self.sample_counts(x, rng)
        names = np.array(self.gene_names + [CONTROL], dtype=object)  # target -1 → names[-1] = control
        obs = pd.DataFrame({"perturbation": names[target], "target": target, "responder": responder})
        return Screen(counts=counts, obs=obs, gene_names=list(self.gene_names))

    # ------------------------------------------------------------------ oracle
    def oracle_delta(self, seed: Any = None, n_cells: int | None = None) -> np.ndarray:
        """Expected Δ of every knockout in *observed* space (normalize + log1p), by Monte Carlo.

        Rows follow ``perturbation_targets``. The draws are shared across conditions (common random
        numbers): the same intrinsic noise, library factors, responders and knockdown sizes. The
        difference therefore reflects the knockout, not resampling. Use it only as ground truth (the
        ceiling, Explain, Discover), never for training.
        """
        cfg = self.config
        rng = np.random.default_rng(seed)
        n = n_cells or cfg.oracle_cells
        eps = rng.standard_normal((n, cfg.n_genes))
        library = rng.lognormal(0.0, cfg.library_sd, size=n)
        kd = np.where(rng.random(n) < cfg.responder_frac, rng.uniform(*cfg.kd_range, size=n), 0.0)

        def expected_expression(x: np.ndarray) -> np.ndarray:
            return normalize_log1p(self.sample_counts(x, rng, library)).mean(axis=0)

        control = expected_expression(self.latent(eps))
        out = np.empty((len(self.perturbation_targets), cfg.n_genes))
        for row, k in enumerate(self.perturbation_targets):
            target = np.full(n, k)
            x = self.add_systematic(self.latent(eps, target, kd), target)
            out[row] = expected_expression(x) - control
        return out

    def disease_program(
        self, seed: Any = None, n_sources: int = 3, n_genes: int = 20
    ) -> tuple[np.ndarray, np.ndarray]:
        """A gene program for the in-silico screen: genes downstream of a few seeded regulators.

        Returns ``(program_genes, source_regulators)``. The screen should rediscover the sources (and
        their upstream regulators) as the knockouts that most suppress the program. Knocked-out genes
        are left out of the program. Their own (masked) self-knockdown must not change the readout.
        """
        rng = np.random.default_rng(seed)
        R = self.config.n_regulators

        def downstream(k: int) -> np.ndarray:
            d = self.descendants(k)
            return d[(d >= R) & ~np.isin(d, self.perturbation_targets)]  # non-regulator, never knocked out

        sizes = np.array([len(downstream(k)) for k in range(R)])
        candidates = np.flatnonzero((sizes >= 10) & (sizes <= 80))
        if len(candidates) < n_sources:
            candidates = np.argsort(-sizes)[: max(n_sources, 1)]
        sources = np.sort(rng.choice(candidates, size=min(n_sources, len(candidates)), replace=False))
        pool = np.unique(np.concatenate([downstream(k) for k in sources]))
        genes = np.sort(rng.choice(pool, size=min(n_genes, len(pool)), replace=False))
        return genes, sources
