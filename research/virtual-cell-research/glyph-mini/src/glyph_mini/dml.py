"""Double/debiased ML for a partially linear logistic model (paper Stage 5).

Model:  logit P(Y=1 | T, W) = theta * T + r(W)
T = the concept, W = confounders, theta = the adjusted log-odds ratio.

Steps (each nuisance is cross-fitted: learned on K-1 folds, predicted on the held-out fold):
  1. pilot logistic Y ~ T + phi(W)          -> r_hat(W), and weights v = mu(1-mu)
  2. weighted regression T ~ phi(W) with v  -> m_hat(W) = E[vT|W]/E[v|W]
  3. T_tilde = T - m_hat(W)                 (the part of the concept W cannot explain)
  4. solve  sum_i (Y_i - sigmoid(theta T_i + r_hat(W_i))) * T_tilde_i = 0  for theta
  5. sandwich SE:  var = mean(psi^2) / (n * J^2),  J = mean(sigmoid'(.) * T * T_tilde)

Glyph uses XGBoost nuisances; here phi(W) = [W, W^2] with linear/logistic models so the
whole thing is readable. The orthogonal score in step 4 is what makes theta robust
to small errors in r_hat and m_hat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from .association import fit_logistic


@dataclass
class DMLResult:
    theta: float
    se: float

    @property
    def odds_ratio(self) -> float:
        return float(np.exp(self.theta))

    @property
    def ci(self) -> tuple[float, float]:
        return float(np.exp(self.theta - 1.96 * self.se)), float(np.exp(self.theta + 1.96 * self.se))

    @property
    def p(self) -> float:
        return float(2 * stats.norm.sf(abs(self.theta) / self.se))


def _phi(W: np.ndarray) -> np.ndarray:
    W = W.reshape(len(W), -1)
    return np.column_stack([np.ones(len(W)), W, W**2])


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def dml_logistic(T: np.ndarray, W: np.ndarray, y: np.ndarray, n_folds: int = 5, seed: int = 0) -> DMLResult:
    n = len(y)
    T = (T - T.mean()) / (T.std() or 1.0)
    folds = np.random.default_rng(seed).permutation(n) % n_folds
    r_hat = np.zeros(n)
    m_hat = np.zeros(n)
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        P_tr, P_te = _phi(W[tr]), _phi(W[te])
        coef, _ = fit_logistic(np.column_stack([T[tr], P_tr]), y[tr], ridge=1e-6)
        r_hat[te] = P_te @ coef[1:]
        mu = _sigmoid(coef[0] * T[tr] + P_tr @ coef[1:])
        v = mu * (1 - mu)
        gamma = np.linalg.lstsq(P_tr * np.sqrt(v)[:, None], T[tr] * np.sqrt(v), rcond=None)[0]
        m_hat[te] = P_te @ gamma
    T_tilde = T - m_hat

    def score(theta: float) -> float:
        return float(np.mean((y - _sigmoid(theta * T + r_hat)) * T_tilde))

    theta = optimize.brentq(score, -10, 10)
    mu = _sigmoid(theta * T + r_hat)
    psi = (y - mu) * T_tilde
    J = np.mean(mu * (1 - mu) * T * T_tilde)
    se = float(np.sqrt(np.mean(psi**2) / (n * J**2)))
    return DMLResult(theta=float(theta), se=se)
