"""Attacks used to measure evasion: in-budget traffic shaping applied to flows a detector flags.

Both attacks run the same forward simulation of padding, dummy packets and delay that the certificate reasons
about, and score every candidate with the detector's exact margin. The random attack draws fixed tries; the
adaptive attack keeps the best operation plans and refines them, so it is much stronger for the same budget.
"""
from __future__ import annotations

import numpy as np

from . import operations as ops


def scorer(ensembles, margin_thresholds, features_list):
    """Distance to the alert line for a detector or a union of detectors: negative means every component calls
    the flow benign (an evasion), so the union takes the largest component value."""
    def score(flows):
        out = None
        for ens, m_thr, feats in zip(ensembles, margin_thresholds, features_list):
            X = np.array([[f_[c] for c in feats] for f_ in flows], dtype=np.float32)
            v = ens.margin(X) - m_thr
            out = v if out is None else np.maximum(out, v)
        return out
    return score


def _flow(z, budget, plan, seed):
    return ops.add_cumulative(ops.apply_operations(z, budget, np.random.default_rng(seed), plan=plan))


def random_attack(z, budget, score, rng, tries=40):
    plans = [{"k": int(rng.integers(0, budget["dummy_pkts"] + 1)), "pad": int(rng.integers(0, budget["pad_bytes"] + 1)),
              "delay": float(rng.uniform(0, budget["delay_ms"]))} for _ in range(tries)]
    flows = [_flow(z, budget, p, int(rng.integers(2 ** 31))) for p in plans]
    s = score(flows)
    j = int(np.argmin(s))
    return float(s[j]), flows[j]


def adaptive_attack(z, budget, score, rng, n_init=60, rounds=4, keep=10, per=6):
    """Start from random plans, then repeatedly keep the best and try neighbours of them (different dummy count,
    padding and delay, and new internal draws). n_init + rounds * keep * per queries in total (300 by default)."""
    K, P, T = budget["dummy_pkts"], budget["pad_bytes"], budget["delay_ms"]
    pool = []
    for _ in range(n_init):
        p = {"k": int(rng.integers(0, K + 1)), "pad": int(rng.integers(0, P + 1)), "delay": float(rng.uniform(0, T))}
        pool.append((p, int(rng.integers(2 ** 31))))
    flows = [_flow(z, budget, p, s) for p, s in pool]
    scores = list(score(flows))
    for _ in range(rounds):
        best = np.argsort(scores)[:keep]
        new = []
        for i in best:
            p, _seed = pool[i]
            for _ in range(per):
                q = {"k": int(np.clip(p["k"] + rng.integers(-1, 2), 0, K)),
                     "pad": int(np.clip(p["pad"] + rng.integers(-15, 16), 0, P)),
                     "delay": float(np.clip(p["delay"] * rng.uniform(0.5, 1.5) + rng.uniform(0, 0.05 * T), 0, T))}
                new.append((q, int(rng.integers(2 ** 31))))
        nf = [_flow(z, budget, p, s) for p, s in new]
        pool += new
        flows += nf
        scores += list(score(nf))
    j = int(np.argmin(scores))
    return float(scores[j]), flows[j]
