"""Learned dynamics models over the shared node space (PyTorch). Same fit/predict contract as models.linear.

GRU      graph-free: per-animal vector of node activities + validity mask -> GRU -> h-step prediction of every node.
TypedGNN typed relational graph state-space model: each node keeps a recurrent state; per step it receives messages
         from chemical presynaptic partners (directed j->i) and gap-junction partners (undirected), each edge type with
         its own message map, every edge scaled by a learned gate g_ij = sigmoid(a_type + b_type * log1p(w_ij) + u_ij)
         (u_ij a free per-edge offset, L1-penalised). Node identity embeddings are shared across animals.
         Gates are predictive edge importances, NOT evidence that a synapse is biologically necessary.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

import os
DEV = os.environ.get("WORMLAB_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def _chunks(data, h, L=64, stride=32):
    xs, ms, ys, ym = [], [], [], []
    for X, M in data:
        T = len(X)
        for s in range(0, T - h - L, stride):
            xs.append(X[s:s + L]); ms.append(M[s:s + L]); ys.append(X[s + h:s + h + L]); ym.append(M[s + h:s + h + L])
    f = lambda a: torch.tensor(np.stack(a), dtype=torch.float32)
    return f(xs), f(ms), f(ys), f(ym)


class _GRUNet(nn.Module):
    def __init__(self, K, hid=128):
        super().__init__()
        self.gru = nn.GRU(2 * K, hid, batch_first=True)
        self.out = nn.Linear(hid, K)

    def forward(self, x, m):
        o, _ = self.gru(torch.cat([x * m, m], -1))
        return self.out(o)


class _GNNNet(nn.Module):
    def __init__(self, K, chem_w, gap_w, d=16, emb=8):
        super().__init__()
        self.K, self.d = K, d
        self.emb = nn.Parameter(0.1 * torch.randn(K, emb))
        self.register_buffer("chem_w", torch.tensor(chem_w, dtype=torch.float32))    # (K, K) row = post
        self.register_buffer("gap_w", torch.tensor(gap_w, dtype=torch.float32))
        self.register_buffer("chem_A", (self.chem_w > 0).float())
        self.register_buffer("gap_A", (self.gap_w > 0).float())
        self.gate_ab = nn.Parameter(torch.zeros(2, 2))                                 # (type, [a, b])
        self.u_chem = nn.Parameter(torch.zeros(K, K)); self.u_gap = nn.Parameter(torch.zeros(K, K))
        self.msg_chem = nn.Linear(d + 2, d); self.msg_gap = nn.Linear(d + 2, d)
        self.cell = nn.GRUCell(2 + emb + 2 * d, d)
        self.out = nn.Linear(d + emb, 1)

    def gates(self):
        gc = torch.sigmoid(self.gate_ab[0, 0] + self.gate_ab[0, 1] * torch.log1p(self.chem_w) + self.u_chem) * self.chem_A
        gg = torch.sigmoid(self.gate_ab[1, 0] + self.gate_ab[1, 1] * torch.log1p(self.gap_w) + self.u_gap) * self.gap_A
        return gc, gg

    def forward(self, x, m):
        B, T, K = x.shape
        hdn = torch.zeros(B, K, self.d, device=x.device)
        gc, gg = self.gates()
        e = self.emb.unsqueeze(0).expand(B, K, -1)
        outs = []
        for t in range(T):
            xt, mt = (x[:, t] * m[:, t]).unsqueeze(-1), m[:, t].unsqueeze(-1)
            src = torch.cat([hdn, xt, mt], -1)                                         # (B, K, d+2)
            mc = torch.einsum("ij,bjd->bid", gc, self.msg_chem(src))
            mg = torch.einsum("ij,bjd->bid", gg, self.msg_gap(src))
            hdn = self.cell(torch.cat([xt, mt, e, mc, mg], -1).reshape(B * K, -1), hdn.reshape(B * K, -1)).reshape(B, K, self.d)
            outs.append(self.out(torch.cat([hdn, e], -1)).squeeze(-1))
        return torch.stack(outs, 1)

    def penalty(self):
        return self.u_chem.abs().mean() + self.u_gap.abs().mean()


class _Trainer:
    name = "learned"

    def __init__(self, epochs=40, lr=2e-3, l1=1e-3, seed=0, batch=32):
        self.epochs, self.lr, self.l1, self.seed, self.batch = epochs, lr, l1, seed, batch

    def _make(self, K):
        raise NotImplementedError

    def fit(self, data, h):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.h = h
        K = data[0][0].shape[1]
        self.net = self._make(K).to(DEV)
        xs, ms, ys, ym = _chunks(data, h)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        n = len(xs)
        for ep in range(self.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, self.batch):
                b = perm[s:s + self.batch]
                x, m, y, w = (a[b].to(DEV) for a in (xs, ms, ys, ym))
                p = self.net(x, m)
                loss = ((p - y) ** 2 * w).sum() / w.sum().clamp(min=1)
                if hasattr(self.net, "penalty"):
                    loss = loss + self.l1 * self.net.penalty()
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def predict(self, X, M):
        self.net.eval()
        x = torch.tensor(X[None], dtype=torch.float32, device=DEV); m = torch.tensor(M[None], dtype=torch.float32, device=DEV)
        P = self.net(x, m)[0].cpu().numpy()
        self.net.train()
        return np.where(M, P, np.nan)          # prediction for t+h needs a valid own input at t (matches linear models)


class GRU(_Trainer):
    name = "gru"

    def _make(self, K):
        return _GRUNet(K)


class TypedGNN(_Trainer):
    name = "typed_gnn"

    def __init__(self, chem_w, gap_w, **kw):
        super().__init__(**kw)
        self.chem_w, self.gap_w = chem_w, gap_w

    def _make(self, K):
        return _GNNNet(K, self.chem_w, self.gap_w)

    def gate_report(self):
        gc, gg = self.net.gates()
        return gc.detach().cpu().numpy(), gg.detach().cpu().numpy()
