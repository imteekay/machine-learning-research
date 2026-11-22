"""Predict: the baselines and the minimal virtual cell.

Every model answers the same question: *how does the cell state change (Δ) when gene k is knocked
down?* The predicted expression is the control state plus Δ.

- ``MeanShift`` predicts the average training response for every knockout.
- ``RidgeDelta`` is the linear baseline: a straight line from the perturbation embedding to Δ.
- ``MinimalVirtualCell`` is an MLP from the perturbation embedding to Δ, built like the blueprint's
  state + perturbation encoder + manipulator.

Any result without the first two baselines is uninterpretable. Several benchmarks found that deep
perturbation models did not beat them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


def masked_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """MSE over the entries of ``target`` that are not NaN (the masked self-knockdown)."""
    mask = ~np.isnan(target)
    return float(np.mean((pred[mask] - target[mask]) ** 2))


class MeanShift:
    """Predict the mean training Δ for every knockout. This is Systema's "perturbed mean".

    It knows that perturbed cells differ from controls (systematic variation), but nothing about
    *which* gene was hit. It is the bar a virtual cell has to clear.
    """

    def fit(self, delta_train: np.ndarray) -> MeanShift:
        self.mean_delta = np.nanmean(delta_train, axis=0)
        return self

    def predict(self, n: int) -> np.ndarray:
        return np.tile(self.mean_delta, (n, 1))


class RidgeDelta:
    """Linear baseline: Δ = mean Δ + (e − mean e) · W, with W fitted by closed-form ridge.

    Inputs and targets are centered, so a huge λ collapses the model to ``MeanShift``. λ is chosen
    on the validation knockouts.
    """

    def __init__(self, lambdas: tuple[float, ...] = tuple(10.0 ** np.arange(-3, 4))):
        self.lambdas = lambdas

    def fit(
        self, emb_train: np.ndarray, delta_train: np.ndarray, emb_val: np.ndarray, delta_val: np.ndarray
    ) -> RidgeDelta:
        self.mean_delta = np.nanmean(delta_train, axis=0)
        self.mean_emb = emb_train.mean(axis=0)
        X = emb_train - self.mean_emb
        Y = np.nan_to_num(delta_train - self.mean_delta)  # masked entries → zero residual
        XtX, XtY, eye = X.T @ X, X.T @ Y, np.eye(X.shape[1])
        best = None
        for lam in self.lambdas:
            W = np.linalg.solve(XtX + lam * eye, XtY)
            score = masked_mse(self._predict(emb_val, W), delta_val)
            if best is None or score < best[0]:
                best = (score, lam, W)
        self.val_mse, self.lam, self.W = best
        return self

    def _predict(self, emb: np.ndarray, W: np.ndarray) -> np.ndarray:
        return self.mean_delta + (emb - self.mean_emb) @ W

    def predict(self, emb: np.ndarray) -> np.ndarray:
        return self._predict(emb, self.W)


class MinimalVirtualCell(nn.Module):
    """The smallest model that deserves the name: a cell state plus a perturbation gives a new state.

    It has three parts, named after the AI virtual cell blueprint (Bunne et al., 2024):

    - ``state``: the control cell state. Here it is the mean control expression, a trivial
      "universal representation". Later it becomes a learned encoder (scVI, a single-cell FM).
    - ``perturbation_encoder``: a fixed table of gene embeddings (``vcell.embeddings``), looked up
      by the knocked-out gene. Because the table covers all genes, unseen knockouts have an input.
    - ``manipulator``: an MLP from the perturbation embedding to Δ expression, a "manipulator
      virtual instrument". Later it gets conditioned on the state and on context (ATAC, E4.1), and
      becomes distributional (flow matching, E4.3).

    The decoder is the identity: expression is predicted directly. Later it becomes an NB likelihood
    (E0.1).

    The output layer starts with zero weights and its bias set to the mean training Δ. Before
    training, the model therefore *is* ``MeanShift``, and training can only add knockout-specific
    signal on top. This is the reference tutorial's fix for collapse. Unlike the tutorial, there is
    no LayerNorm: it would rescale every sample and erase the embedding norm, which here encodes how
    strongly a gene is wired into the network.
    """

    def __init__(
        self,
        gene_embeddings: np.ndarray,
        ctrl_state: np.ndarray,
        hidden: int = 64,
        init_delta: np.ndarray | None = None,
    ):
        super().__init__()
        n_genes, emb_dim = gene_embeddings.shape
        self.register_buffer("state", torch.as_tensor(ctrl_state, dtype=torch.float32))
        self.register_buffer(
            "perturbation_encoder", torch.as_tensor(gene_embeddings, dtype=torch.float32)
        )
        self.manipulator = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.GELU(), nn.Linear(hidden, len(ctrl_state))
        )
        last = self.manipulator[-1]
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.copy_(
                torch.zeros(len(ctrl_state))
                if init_delta is None
                else torch.as_tensor(init_delta, dtype=torch.float32)
            )

    def delta(self, gene_index: torch.Tensor) -> torch.Tensor:
        """Predicted change in expression when the genes in ``gene_index`` are knocked down."""
        return self.manipulator(self.perturbation_encoder[gene_index])

    def forward(self, gene_index: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        """Predicted expression after the knockdown: state + Δ.

        Pass single control cells as ``state`` to shift a whole population.
        """
        return (self.state if state is None else state) + self.delta(gene_index)

    @torch.no_grad()
    def predict_delta(self, gene_index: np.ndarray) -> np.ndarray:
        self.eval()
        index = torch.as_tensor(np.atleast_1d(gene_index), dtype=torch.long)
        return self.delta(index).numpy().astype(float)


@dataclass(frozen=True)
class FitConfig:
    hidden: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 2000
    patience: int = 100

    def to_dict(self) -> dict:
        return asdict(self)


def _masked_target(delta: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    mask = ~np.isnan(delta)
    return (
        torch.as_tensor(np.nan_to_num(delta), dtype=torch.float32),
        torch.as_tensor(mask, dtype=torch.float32),
    )


def _masked_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (((pred - target) ** 2) * mask).sum() / mask.sum()


def fit_virtual_cell(
    gene_embeddings: np.ndarray,
    ctrl_state: np.ndarray,
    train_targets: np.ndarray,
    delta_train: np.ndarray,
    val_targets: np.ndarray,
    delta_val: np.ndarray,
    seed: int = 0,
    config: FitConfig | None = None,
) -> MinimalVirtualCell:
    """Train with full-batch AdamW on the masked MSE of Δ, then early-stop on the validation knockouts.

    The best weights are restored, and the untrained model (= MeanShift) counts as a candidate. If
    nothing generalizes, you get the baseline back instead of an overfitted model. Training is
    deterministic: CPU, one thread, and a fixed seed.
    """
    cfg = config or FitConfig()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = MinimalVirtualCell(
        gene_embeddings, ctrl_state, hidden=cfg.hidden, init_delta=np.nanmean(delta_train, axis=0)
    )
    x_train = torch.as_tensor(train_targets, dtype=torch.long)
    x_val = torch.as_tensor(val_targets, dtype=torch.long)
    y_train, m_train = _masked_target(delta_train)
    y_val, m_val = _masked_target(delta_val)
    optimizer = torch.optim.AdamW(model.manipulator.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def snapshot() -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in model.manipulator.state_dict().items()}

    model.eval()
    with torch.no_grad():
        best_val = _masked_loss(model.delta(x_val), y_val, m_val).item()
    best_state, best_epoch, waited = snapshot(), 0, 0
    history = [(0, float("nan"), best_val)]
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = _masked_loss(model.delta(x_train), y_train, m_train)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val = _masked_loss(model.delta(x_val), y_val, m_val).item()
        history.append((epoch, loss.item(), val))
        if val < best_val:
            best_val, best_state, best_epoch, waited = val, snapshot(), epoch, 0
        else:
            waited += 1
            if waited >= cfg.patience:
                break
    model.manipulator.load_state_dict(best_state)
    model.best_epoch = best_epoch
    model.history = history
    return model
