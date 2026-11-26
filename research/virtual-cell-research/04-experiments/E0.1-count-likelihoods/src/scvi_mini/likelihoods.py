"""Log-probability mass functions on UMI counts, all in the same space so they are comparable.

- Poisson(mu): variance = mu. The "pure sampling noise" model of droplet capture.
- NB(mu, theta): variance = mu + mu^2/theta. Adds biological/technical overdispersion.
- ZINB(mu, theta, pi): extra point mass at zero ("dropout").
- Discretised Gaussian on log1p(CP10k): the usual Gaussian-on-log-data model, turned into a proper
  PMF on counts by integrating its density over each count's bin [k-0.5, k+0.5] after the
  monotone transform g(k) = log1p(k * 1e4 / L). Without this, a density and a mass cannot be
  compared, and the Gaussian would look better or worse purely by units.
"""

from __future__ import annotations

import math

import torch
from torch.special import log_ndtr

EPS = 1e-8


def poisson_logpmf(x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    return x * torch.log(mu + EPS) - mu - torch.lgamma(x + 1)


def nb_logpmf(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    log_t_mu = torch.log(theta + mu + EPS)
    return (torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1)
            + theta * (torch.log(theta + EPS) - log_t_mu) + x * (torch.log(mu + EPS) - log_t_mu))


def zinb_logpmf(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, pi_logits: torch.Tensor) -> torch.Tensor:
    nb = nb_logpmf(x, mu, theta)
    log_pi = torch.nn.functional.logsigmoid(pi_logits)
    log_1mpi = torch.nn.functional.logsigmoid(-pi_logits)
    nb_zero = theta * (torch.log(theta + EPS) - torch.log(theta + mu + EPS))
    zero = torch.logaddexp(log_pi, log_1mpi + nb_zero)
    return torch.where(x < 0.5, zero, log_1mpi + nb)


def _log_diff_ndtr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """log(Phi(b) - Phi(a)) for b > a, stable in both tails (reflect when a > 0)."""
    flip = a > 0
    a2, b2 = torch.where(flip, -b, a), torch.where(flip, -a, b)
    lb, la = log_ndtr(b2), log_ndtr(a2)
    return lb + torch.log(-torch.expm1(torch.clamp(la - lb, max=-1e-12)))


def gaussian_log1p_logpmf(x: torch.Tensor, mean: torch.Tensor, sd: torch.Tensor, lib_total: torch.Tensor) -> torch.Tensor:
    """P(X = k) under y = log1p(X * 1e4 / L) ~ N(mean, sd), integrated over the count's bin."""
    s = 1e4 / lib_total
    hi = torch.log1p((x + 0.5) * s)
    lo = torch.where(x < 0.5, torch.full_like(x, -math.inf), torch.log1p(torch.clamp(x - 0.5, min=0) * s))
    a = (lo - mean) / sd
    b = (hi - mean) / sd
    return _log_diff_ndtr(a, b)


def gaussian_logpdf(y: torch.Tensor, mean: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    return -0.5 * ((y - mean) / sd) ** 2 - torch.log(sd) - 0.5 * math.log(2 * math.pi)
