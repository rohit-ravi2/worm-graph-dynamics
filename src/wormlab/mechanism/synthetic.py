"""Synthetic data provider: known-ground-truth cohorts for DISCOVERY and CONFIRMATION partitions.

Payload tables are only built inside ``table()``, which requires a broker grant, so the firewall is exercised
exactly as for real data.
"""
from __future__ import annotations

from . import generators as G
from . import observations as O
from .engine import DataProvider


class SyntheticProvider(DataProvider):
    def __init__(self, *, seed: int, n_discovery: int = 8, n_confirmation: int = 10, drop_labels: bool = False, n_nuclei: int = 150,
                 conn_kw: dict | None = None, **params):
        self.seed, self.params, self.drop_labels = seed, params, drop_labels
        self.sizes = {"DISCOVERY": n_discovery, "CONFIRMATION": n_confirmation}
        self.n_nuclei, self.conn_kw = n_nuclei, conn_kw or {}
        self._cache = {}

    def units(self, partition):
        n = self.sizes.get(partition, 0)
        return [f"{partition.lower()}-syn-{i}" for i in range(n)]

    def data_stream(self):
        return f"synthetic:{self.seed}"

    def table(self, partition, grant):
        if not grant.get("granted"):
            raise PermissionError("table() requires a granted broker entry")
        if partition not in self._cache:
            offset = {"DISCOVERY": 0, "CONFIRMATION": 1000}[partition]
            coh = G.generate_cohort(self.seed + offset, self.sizes[partition], n_nuclei=self.n_nuclei, conn_kw=self.conn_kw, **self.params)
            tabs = [O.build_table(p, f"{partition.lower()}-{a}", None if self.drop_labels else conn) for a, p, conn in coh]
            self._cache[partition] = O.concat(tabs)
        return self._cache[partition]
