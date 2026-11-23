"""Ensemble of small MLP dynamics models predicting the state delta.

Each member maps normalised [state, action] -> normalised (next_state - state). Members are
evaluated together with batched matmuls (weights have a leading ensemble dim), so a planner
can roll out ``E x N`` imagined trajectories in one call. Ensemble disagreement (std across
members) is used as the uncertainty signal.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from envs.push_env import ACT_DIM, STATE_DIM


class EnsembleLinear(nn.Module):
    def __init__(self, n_members: int, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_members, d_in, d_out))
        self.bias = nn.Parameter(torch.zeros(n_members, 1, d_out))
        bound = 1.0 / math.sqrt(d_in)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):  # x: (E, N, d_in)
        return torch.baddbmm(self.bias, x, self.weight)


class DynamicsEnsemble(nn.Module):
    def __init__(self, n_members: int = 5, hidden: int = 200, layers: int = 3,
                 state_dim: int = STATE_DIM, act_dim: int = ACT_DIM):
        super().__init__()
        self.n_members = n_members
        self.state_dim, self.act_dim = state_dim, act_dim
        d_in = state_dim + act_dim
        dims = [d_in] + [hidden] * layers + [state_dim]
        self.layers = nn.ModuleList(EnsembleLinear(n_members, a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.act = nn.SiLU()
        for name in ("in_mu", "in_sd", "out_mu", "out_sd"):
            d = d_in if name.startswith("in") else state_dim
            self.register_buffer(name, torch.zeros(d) if "mu" in name else torch.ones(d))

    # ---------------------------------------------------------------- core
    def _forward_norm(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.act(h)
        return h

    def forward(self, state, action):
        """state: (E, N, S), action: (E, N, A) -> next_state (E, N, S)."""
        x = (torch.cat([state, action], -1) - self.in_mu) / self.in_sd
        delta = self._forward_norm(x) * self.out_sd + self.out_mu
        return state + delta

    @torch.no_grad()
    def rollout(self, state0: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Open-loop imagined rollouts for every member.

        state0: (S,) or (N, S); actions: (N, H, A). Returns (E, N, H+1, S) incl. state0.
        """
        acts = torch.as_tensor(actions, dtype=torch.float32)
        N, H, _ = acts.shape
        S = self.state_dim
        s = torch.as_tensor(np.broadcast_to(state0, (N, S)).copy(), dtype=torch.float32)
        s = s.unsqueeze(0).expand(self.n_members, N, S).contiguous()
        out = [s]
        a_e = acts.unsqueeze(0).expand(self.n_members, N, H, self.act_dim)
        for t in range(H):
            s = self.forward(s, a_e[:, :, t])
            out.append(s)
        return torch.stack(out, 2).numpy()

    # ---------------------------------------------------------------- training
    def set_normalizer(self, s, a, ns):
        x = np.concatenate([s, a], -1)
        d = ns - s
        f = lambda v: torch.as_tensor(v, dtype=torch.float32)
        self.in_mu.copy_(f(x.mean(0)))
        self.in_sd.copy_(f(x.std(0) + 1e-6))
        self.out_mu.copy_(f(d.mean(0)))
        self.out_sd.copy_(f(d.std(0) + 1e-6))

    def fit(self, s, a, ns, epochs_steps: int = 3000, batch: int = 256, lr: float = 1e-3,
            weights: np.ndarray | None = None, seed: int = 0, refit_normalizer: bool = True,
            verbose: bool = False) -> float:
        """Train every member on (bootstrap / error-weighted) minibatches. Returns final loss."""
        if refit_normalizer:
            self.set_normalizer(s, a, ns)
        g = torch.Generator().manual_seed(seed)
        S = torch.as_tensor(s, dtype=torch.float32)
        A = torch.as_tensor(a, dtype=torch.float32)
        D = (torch.as_tensor(ns - s, dtype=torch.float32) - self.out_mu) / self.out_sd
        X = (torch.cat([S, A], -1) - self.in_mu) / self.in_sd
        n = len(S)
        p = None
        if weights is not None:
            p = torch.as_tensor(weights / weights.sum(), dtype=torch.float32)
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        loss_val = float("nan")
        for it in range(epochs_steps):
            if p is None:
                idx = torch.randint(0, n, (self.n_members, batch), generator=g)
            else:
                idx = torch.multinomial(p, self.n_members * batch, replacement=True, generator=g).view(self.n_members, batch)
            pred = self._forward_norm(X[idx])
            loss = ((pred - D[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_val = float(loss.detach())
            if verbose and it % 1000 == 0:
                print(f"  wm it {it} loss {loss_val:.4f}", flush=True)
        return loss_val

    @torch.no_grad()
    def one_step_error(self, s, a, ns) -> np.ndarray:
        """Per-transition error of the ensemble mean (unnormalised L2 over the full state)."""
        S = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0).expand(self.n_members, -1, -1)
        A = torch.as_tensor(a, dtype=torch.float32).unsqueeze(0).expand(self.n_members, -1, -1)
        pred = self.forward(S, A).mean(0).numpy()
        return np.linalg.norm(pred - ns, axis=-1)

    def save(self, path: Path):
        torch.save(self.state_dict(), path)

    def load(self, path: Path):
        self.load_state_dict(torch.load(path, map_location="cpu"))
        return self
