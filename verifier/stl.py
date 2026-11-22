"""A small, vectorised Signal Temporal Logic (STL) subset with quantitative robustness.

Formulas are evaluated over discrete-time signals. Every node returns a robustness
*signal* of shape ``(..., T)`` (leading batch dims allowed), so the same code scores one
real trajectory or thousands of imagined ones at once.

Supported: predicates, ``Not``, ``And``, ``Or``, ``Always[a,b]``, ``Eventually[a,b]``.
Semantics (standard, finite-trace, windows clipped at the end of the trace):

    rho(p, t)                 = f_p(x_t)
    rho(not phi, t)           = -rho(phi, t)
    rho(phi and psi, t)       = min(rho(phi, t), rho(psi, t))
    rho(phi or psi, t)        = max(rho(phi, t), rho(psi, t))
    rho(G[a,b] phi, t)        = min_{t' in [t+a, min(t+b, T-1)]} rho(phi, t')
    rho(F[a,b] phi, t)        = max_{t' in [t+a, min(t+b, T-1)]} rho(phi, t')

If a window is empty (t + a > T-1), G returns +inf and F returns -inf (vacuous truth /
unsatisfiable), matching the usual finite-trace convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Signal = np.ndarray  # shape (..., T)


class Formula:
    def robustness_signal(self, ctx) -> Signal:
        raise NotImplementedError

    def robustness(self, ctx) -> np.ndarray:
        """Robustness at time 0, shape ``(...)``."""
        return self.robustness_signal(ctx)[..., 0]


@dataclass
class Predicate(Formula):
    name: str
    fn: Callable[[object], Signal]

    def robustness_signal(self, ctx):
        return self.fn(ctx)

    def __str__(self):
        return self.name


@dataclass
class Not(Formula):
    phi: Formula

    def robustness_signal(self, ctx):
        return -self.phi.robustness_signal(ctx)

    def __str__(self):
        return f"not ({self.phi})"


@dataclass
class And(Formula):
    children: list

    def robustness_signal(self, ctx):
        return np.minimum.reduce([c.robustness_signal(ctx) for c in self.children])

    def __str__(self):
        return " and ".join(f"({c})" for c in self.children)


@dataclass
class Or(Formula):
    children: list

    def robustness_signal(self, ctx):
        return np.maximum.reduce([c.robustness_signal(ctx) for c in self.children])

    def __str__(self):
        return " or ".join(f"({c})" for c in self.children)


def _window_reduce(sig: Signal, a: int, b: int, reduce: str) -> Signal:
    """Windowed min/max over [t+a, t+b] for every t, clipped at the trace end."""
    T = sig.shape[-1]
    fill = np.inf if reduce == "min" else -np.inf
    out = np.full(sig.shape, fill, dtype=np.float64)
    # Reverse running reduction is O(T * window); T is small (<= ~100) so this is fine.
    for t in range(T):
        lo, hi = t + a, min(t + b, T - 1)
        if lo > hi:
            continue
        out[..., t] = (sig[..., lo:hi + 1].min(axis=-1) if reduce == "min"
                       else sig[..., lo:hi + 1].max(axis=-1))
    return out


@dataclass
class Always(Formula):
    phi: Formula
    a: int = 0
    b: int = 10**9

    def robustness_signal(self, ctx):
        return _window_reduce(self.phi.robustness_signal(ctx), self.a, self.b, "min")

    def robustness(self, ctx):
        # Fast path for evaluation at t = 0 only.
        sig = self.phi.robustness_signal(ctx)
        hi = min(self.b, sig.shape[-1] - 1)
        if self.a > hi:
            return np.full(sig.shape[:-1], np.inf)
        return sig[..., self.a:hi + 1].min(axis=-1)

    def __str__(self):
        return f"G[{self.a},{self.b}] ({self.phi})"


@dataclass
class Eventually(Formula):
    phi: Formula
    a: int = 0
    b: int = 10**9

    def robustness_signal(self, ctx):
        return _window_reduce(self.phi.robustness_signal(ctx), self.a, self.b, "max")

    def robustness(self, ctx):
        sig = self.phi.robustness_signal(ctx)
        hi = min(self.b, sig.shape[-1] - 1)
        if self.a > hi:
            return np.full(sig.shape[:-1], -np.inf)
        return sig[..., self.a:hi + 1].max(axis=-1)

    def __str__(self):
        return f"F[{self.a},{self.b}] ({self.phi})"
