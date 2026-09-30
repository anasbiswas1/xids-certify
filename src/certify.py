"""Deployment-time certificates for XGBoost flow detectors.

For a flow the detector calls benign, decide whether any flow in its in-budget preimage (every flow the declared
attacker operations could have turned into it) would have been flagged. Three outcomes: CERTIFIED (proof that none
would), WITNESS (a concrete preimage the detector flags, checked with the real model), UNRESOLVED (no proof within
the time limit, or a candidate that failed the check). Every relaxation widens the preimage, so a certificate is
never issued for a flow that has a flagged preimage inside the model.
"""
from __future__ import annotations

import json
import time

import numpy as np

STATUS = ("CERTIFIED", "WITNESS", "UNRESOLVED")


class Ensemble:
    """Trees of a binary:logistic booster, read from the model's own JSON (exact float32 split values)."""

    def __init__(self, booster, features):
        raw = json.loads(booster.save_raw(raw_format="json"))
        names = raw["learner"].get("feature_names") or []
        if list(names) != list(features):
            raise ValueError("booster feature names differ from the detector feature list")
        self.features = list(features)
        self.trees = []
        for t in raw["learner"]["gradient_booster"]["model"]["trees"]:
            left = np.asarray(t["left_children"], dtype=np.int64)
            right = np.asarray(t["right_children"], dtype=np.int64)
            feat = np.asarray(t["split_indices"], dtype=np.int64)
            cond = np.asarray(t["split_conditions"], dtype=np.float32)
            self.trees.append((left, right, feat, cond))
        self.base = 0.0
        self.base = self._calibrate_base(booster)

    def leaf_sum(self, X32: np.ndarray) -> np.ndarray:
        """Sum of leaf values with XGBoost's rule: go left when x < split (float32 comparison)."""
        out = np.zeros(len(X32), dtype=np.float64)
        for left, right, feat, cond in self.trees:
            node = np.zeros(len(X32), dtype=np.int64)
            active = left[node] != -1
            while active.any():
                idx = np.flatnonzero(active)
                n = node[idx]
                go_left = X32[idx, feat[n]] < cond[n]
                node[idx] = np.where(go_left, left[n], right[n])
                active = left[node] != -1
            out += cond[node].astype(np.float64)
        return out

    def _calibrate_base(self, booster, n=2000, seed=0):
        import xgboost as xgb
        rng = np.random.default_rng(seed)
        thr = [c[l != -1] for l, r, f, c in self.trees]
        pool = np.concatenate([t for t in thr if len(t)]) if any(len(t) for t in thr) else np.zeros(1, np.float32)
        X = rng.choice(pool, size=(n, len(self.features))).astype(np.float32)
        X += rng.choice([-1.0, 0.0, 1.0], size=X.shape).astype(np.float32)
        m = booster.predict(xgb.DMatrix(X, feature_names=self.features), output_margin=True).astype(np.float64)
        diff = m - self.leaf_sum(X)
        if np.ptp(diff) > 1e-3:
            raise RuntimeError(f"tree parser disagrees with the booster (spread {np.ptp(diff):.3g})")
        return float(np.median(diff))

    def margin(self, X32: np.ndarray) -> np.ndarray:
        return self.base + self.leaf_sum(X32)


# ------------------------------------------------------------------ preimage model + exact solver
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from . import operations as ops

TOL_BR = 1e-6          # bracket tolerance, as in the relation audit
FEAS_EPS = 1e-6        # certificate asks for no preimage with margin >= threshold - FEAS_EPS (conservative)
LATE_GRACE = 0.05      # seconds of scheduling slack before an answer counts as late


def logit(p):
    return float(np.log(p) - np.log1p(-p))


class _LP:
    def __init__(self):
        self.lb, self.ub, self.integ, self.rows = [], [], [], []

    def var(self, lb, ub, integer=False):
        self.lb.append(float(lb))
        self.ub.append(float(ub))
        self.integ.append(1 if integer else 0)
        return len(self.lb) - 1

    def row(self, coefs, lo, hi, tag):
        self.rows.append((coefs, float(lo), float(hi), tag))


def out_of_model(x: dict) -> list:
    """Audited relations that the observed flow itself breaks; such a flow cannot be certified."""
    bad = []
    dur, din, dout = x["FLOW_DURATION_MILLISECONDS"], x["DURATION_IN"], x["DURATION_OUT"]
    if din > dur + 1 or dout > dur + 1:
        bad.append("duration")
    hist = sum(x[b] for b in ops.HIST)
    if hist < x["IN_PKTS"] + x["OUT_PKTS"]:
        bad.append("hist_sum_ge_all_packets")
    if x["LONGEST_FLOW_PKT"] != x["MAX_IP_PKT_LEN"] or x["LONGEST_FLOW_PKT"] > ops.MTU:
        bad.append("longest")
    if x["SHORTEST_FLOW_PKT"] < x["MIN_IP_PKT_LEN"] - 1:
        bad.append("shortest_ge_min_ip_len")
    if x["OUT_PKTS"] > 0 and not (x["OUT_PKTS"] * x["SHORTEST_FLOW_PKT"] <= x["OUT_BYTES"] + 1
                                  <= x["OUT_PKTS"] * x["LONGEST_FLOW_PKT"] + 2):
        bad.append("out_bytes_within_pkt_bounds")
    if x["RETRANSMITTED_IN_PKTS"] > x["IN_PKTS"] or x["RETRANSMITTED_IN_BYTES"] > x["IN_BYTES"] + 1:
        bad.append("retrans_in_le_totals")
    if int(x["TCP_FLAGS"]) != (int(x["CLIENT_TCP_FLAGS"]) | int(x["SERVER_TCP_FLAGS"])):
        bad.append("tcp_flags_or")
    for f, byts in (("SRC_TO_DST_AVG_THROUGHPUT", x["IN_BYTES"]), ("DST_TO_SRC_AVG_THROUGHPUT", x["OUT_BYTES"])):
        if dur > 0:
            lo, hi = 8000.0 * byts / (dur + 1), 8000.0 * byts / dur
            if not (lo * (1 - TOL_BR) - 1 <= x[f] <= hi * (1 + TOL_BR) + 1):
                bad.append(f"bracket_{f}")
    return bad


def _feature_bounds(x, budget, big):
    """Widest range each changeable feature of the preimage can take (operations used to the full)."""
    P, K, T = budget["pad_bytes"], budget["dummy_pkts"], budget["delay_ms"]
    keff = int(max(0, min(K, x["IN_PKTS"] - max(1.0, x["RETRANSMITTED_IN_PKTS"]))))
    b = {}
    b["IN_PKTS"] = (x["IN_PKTS"] - keff, x["IN_PKTS"])
    b["IN_BYTES"] = (max(0.0, x["IN_BYTES"] - P * x["IN_PKTS"] - ops.MTU * keff), x["IN_BYTES"])
    b["RETRANSMITTED_IN_BYTES"] = (max(0.0, x["RETRANSMITTED_IN_BYTES"] - P * x["RETRANSMITTED_IN_PKTS"]),
                                   x["RETRANSMITTED_IN_BYTES"])
    for f in ("FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT"):
        b[f] = (max(0.0, x[f] - T), x[f])
    lng_lo = 0.0 if keff else max(0.0, x["LONGEST_FLOW_PKT"] - P)
    b["LONGEST_FLOW_PKT"] = b["MAX_IP_PKT_LEN"] = (lng_lo, x["LONGEST_FLOW_PKT"])
    for f in ("SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN"):
        b[f] = (max(0.0, x[f] - P), float(ops.MTU) if keff else x[f])
    if x["OUT_PKTS"] > 0:
        cap = (x["OUT_BYTES"] + 1) / x["OUT_PKTS"]
        for f in ("SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN"):
            b[f] = (b[f][0], max(b[f][0], min(b[f][1], cap)))
        floor_ = (x["OUT_BYTES"] - 1) / x["OUT_PKTS"]
        for f in ("LONGEST_FLOW_PKT", "MAX_IP_PKT_LEN"):
            b[f] = (min(b[f][1], max(b[f][0], floor_)), b[f][1])
    total = sum(x[h] for h in ops.HIST)
    for h in ops.HIST:
        b[h] = (0.0, total)
    for f in ops.IAT + ops.SECOND_BYTES:
        b[f] = (0.0, max(x[f], big.get(f, 0.0)))
    b["CLIENT_TCP_FLAGS"] = (0.0 if keff else x["CLIENT_TCP_FLAGS"], x["CLIENT_TCP_FLAGS"])
    b["TCP_FLAGS"] = (x["SERVER_TCP_FLAGS"] if keff else x["TCP_FLAGS"], x["TCP_FLAGS"])
    b["TCP_WIN_MAX_IN"] = (0.0 if keff else x["TCP_WIN_MAX_IN"], x["TCP_WIN_MAX_IN"])
    return b, keff


def _thr_box(x, b, name):
    dlo, dhi = b["FLOW_DURATION_MILLISECONDS"]
    if name == "SRC_TO_DST_AVG_THROUGHPUT":
        blo, bhi = b["IN_BYTES"]
    else:
        blo = bhi = x["OUT_BYTES"]
    lo = 8000.0 * blo * (1 - TOL_BR) / (dhi + 1) - 1
    hi = (8000.0 * bhi * (1 + TOL_BR) / dlo + 1) if dlo > 0 else np.inf
    return min(lo, x[name]), max(hi, x[name])


class Verifier:
    """Certificate check for one detector: trees, decision threshold, integer features, budget."""

    def __init__(self, ens: Ensemble, threshold: float, integer_feats, budget: dict, maximise: bool = False,
                 search_share: float = 0.35, seed: int = 20260930):
        self.ens, self.budget, self.maximise = ens, dict(budget), bool(maximise)
        self.search_share, self.rng = float(search_share), np.random.default_rng(seed)
        self.m_thr = logit(threshold)
        self.threshold = float(threshold)
        self.F = ens.features
        self.fi = {f: i for i, f in enumerate(self.F)}
        self.integer = set(integer_feats)
        self.big, self.cuts = {}, {}
        for f in self.F:
            j = self.fi[f]
            ts = [c[(l != -1) & (fe == j)] for l, r, fe, c in ens.trees]
            ts = np.unique(np.concatenate(ts)) if ts else np.zeros(0, np.float32)
            self.cuts[f] = ts.astype(np.float64)
            self.big[f] = float(ts.max()) * 1.01 + 1.0 if len(ts) else 1.0
        missing = [f for f in ops.UNCHANGED + ops.FEATURE_ROLES["exact"] + ops.FEATURE_ROLES["range"]
                   + ops.THROUGHPUT + ops.IAT + ops.SECOND_BYTES if f not in self.fi]
        if missing:
            raise ValueError(f"operation model names features the detector lacks: {missing}")

    # ---------------------------------------------------------------- box bound
    def _box(self, x):
        b, keff = _feature_bounds(x, self.budget, self.big)
        lo = np.array([x[f] for f in self.F], dtype=np.float64)
        hi = lo.copy()
        for f, (l, h) in b.items():
            lo[self.fi[f]], hi[self.fi[f]] = l, h
        for f in ops.THROUGHPUT:
            lo[self.fi[f]], hi[self.fi[f]] = _thr_box(x, b, f)
        return lo, hi, b, keff

    def _reach(self, tree, lo, hi):
        left, right, feat, cond = tree
        leaves, internal, stack = [], [], [0]
        while stack:
            n = stack.pop()
            if left[n] == -1:
                leaves.append(n)
                continue
            f, t = feat[n], float(cond[n])
            gl, gr = lo[f] < t, hi[f] >= t
            if gl and gr:
                internal.append(n)
            if gl:
                stack.append(left[n])
            if gr:
                stack.append(right[n])
        return leaves, internal

    # ---------------------------------------------------------------- fast witness search
    def _cands(self, f, a, b, cur, n_max=48):
        """Values of feature f in [a, b] that land in every split interval the trees distinguish."""
        if b < a:
            return np.array([cur])
        t = self.cuts[f]
        t = t[(t > a) & (t <= b)]
        if f in self.integer:
            vals = np.concatenate([[a, b, cur], np.ceil(t), np.ceil(t) - 1])
            vals = np.round(vals)
        else:
            t32 = t.astype(np.float32)
            vals = np.concatenate([[a, b, cur], t, np.nextafter(t32, np.float32(-np.inf)).astype(np.float64)])
        vals = np.unique(np.clip(vals, a, b))
        if len(vals) > n_max:
            keep = np.linspace(0, len(vals) - 1, n_max).round().astype(int)
            vals = np.unique(np.concatenate([vals[keep], [cur]]))
        return vals

    def _bracket(self, byts, dur):
        if dur <= 0:
            return 8000.0 * byts, 8000.0 * byts
        return 8000.0 * byts / (dur + 1), 8000.0 * byts / dur

    def _interval(self, f, z, o, x, bnd):
        P, T = self.budget["pad_bytes"], self.budget["delay_ms"]
        u, v, d = o["u"], o["v"], o["d"]
        lo_b, hi_b = bnd.get(f, (x[f], x[f]))
        if f in ops.IAT:
            return (lo_b, hi_b) if v else (x[f], x[f])
        if f in ops.SECOND_BYTES:
            return (lo_b, hi_b) if u else (x[f], x[f])
        if f == "LONGEST_FLOW_PKT":
            return max(lo_b, x[f] - P * u - ops.MTU * d), x[f]
        if f == "SHORTEST_FLOW_PKT":
            return max(lo_b, x[f] - P * u, z["MIN_IP_PKT_LEN"] - 1), min(hi_b, x[f] + ops.MTU * d)
        if f == "MIN_IP_PKT_LEN":
            return max(lo_b, x[f] - P * u), min(hi_b, x[f] + ops.MTU * d, z["SHORTEST_FLOW_PKT"] + 1)
        if f == "FLOW_DURATION_MILLISECONDS":
            return max(lo_b, x[f] - T * v, z["DURATION_IN"] - 1, z["DURATION_OUT"] - 1), x[f]
        if f in ("DURATION_IN", "DURATION_OUT"):
            return max(lo_b, x[f] - T * v), min(x[f], z["FLOW_DURATION_MILLISECONDS"] + 1)
        if f == "IN_BYTES":
            k = o["k"]
            lo = max(0.0, x[f] - P * (x["IN_PKTS"] - k) * u - ops.MTU * k, z["RETRANSMITTED_IN_BYTES"] - 1)
            return lo, x[f] - ops.DUMMY_MIN * k
        if f == "RETRANSMITTED_IN_BYTES":
            return max(lo_b, x[f] - P * x["RETRANSMITTED_IN_PKTS"] * u), min(x[f], z["IN_BYTES"] + 1)
        if f == "TCP_WIN_MAX_IN":
            return (0.0, x[f]) if d else (x[f], x[f])
        return x[f], x[f]

    def _set(self, z, f, val, o, x):
        z[f] = float(val)
        if f == "LONGEST_FLOW_PKT":
            z["MAX_IP_PKT_LEN"] = float(val)
        if f in ("IN_BYTES", "FLOW_DURATION_MILLISECONDS"):
            for name, byts, gate in (("SRC_TO_DST_AVG_THROUGHPUT", z["IN_BYTES"], o["u"]),
                                     ("DST_TO_SRC_AVG_THROUGHPUT", x["OUT_BYTES"], o["v"])):
                if gate:
                    lo, hi = self._bracket(byts, z["FLOW_DURATION_MILLISECONDS"])
                    z[name] = float(min(max(z[name], lo), hi))

    def _vec(self, z):
        return np.array([z[f] for f in self.F], dtype=np.float32)

    def _starts(self, x, keff):
        P, T = self.budget["pad_bytes"], self.budget["delay_ms"]
        out = []
        for k in sorted({0, keff, max(0, keff // 2)}):
            for u, v in ((1, 1), (1, 0)):
                if (v == 0 and k > 0) or (v and not (T > 0 or k > 0)) or (u and not (P > 0 or T > 0 or k > 0)):
                    continue
                out.append({"k": k, "d": int(k > 0), "u": u, "v": v})
        return out or [{"k": 0, "d": 0, "u": 0, "v": 0}]

    def _search(self, x, bnd, keff, deadline):
        feats = (ops.IAT + ops.SECOND_BYTES + ["SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "LONGEST_FLOW_PKT",
                 "IN_BYTES", "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT", "RETRANSMITTED_IN_BYTES",
                 "TCP_WIN_MAX_IN", "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT", "CLIENT_TCP_FLAGS"])
        for o in self._starts(x, keff):
            z = dict(x)
            z["IN_PKTS"] = x["IN_PKTS"] - o["k"]
            if o["k"]:
                self._set(z, "IN_BYTES", x["IN_BYTES"] - ops.DUMMY_MIN * o["k"], o, x)
            cur = float(self.ens.margin(self._vec(z)[None, :])[0])
            for _sweep in range(3):
                improved = False
                for f in feats:
                    if time.perf_counter() > deadline:
                        return None
                    if f == "CLIENT_TCP_FLAGS":
                        if not o["d"]:
                            continue
                        bits = [1 << i for i in range(9) if int(x[f]) & (1 << i)]
                        vals = sorted({sum(bb for i, bb in enumerate(bits) if m >> i & 1) for m in range(2 ** len(bits))})
                    elif f in ops.THROUGHPUT:
                        gate = o["u"] if f == "SRC_TO_DST_AVG_THROUGHPUT" else o["v"]
                        if not gate:
                            continue
                        byts = z["IN_BYTES"] if f == "SRC_TO_DST_AVG_THROUGHPUT" else x["OUT_BYTES"]
                        a, b_ = self._bracket(byts, z["FLOW_DURATION_MILLISECONDS"])
                        vals = self._cands(f, a * (1 - TOL_BR), b_, z[f])
                        vals = vals[(vals >= a * (1 - TOL_BR) - 1e-9) & (vals <= b_ * (1 + TOL_BR) + 1e-9)]
                        if not len(vals):
                            continue
                    else:
                        a, b_ = self._interval(f, z, o, x, bnd)
                        if b_ <= a:
                            continue
                        vals = self._cands(f, a, b_, z[f])
                    Z = []
                    for val in vals:
                        zz = dict(z)
                        if f == "CLIENT_TCP_FLAGS":
                            zz[f] = float(val)
                            zz["TCP_FLAGS"] = float(int(val) | int(x["SERVER_TCP_FLAGS"]))
                        else:
                            self._set(zz, f, val, o, x)
                        Z.append(zz)
                    m = self.ens.margin(np.stack([self._vec(q) for q in Z]))
                    j = int(np.argmax(m))
                    if m[j] > cur + 1e-12:
                        z, cur, improved = Z[j], float(m[j]), True
                    if cur >= self.m_thr:
                        found = self._confirm(x, z, o, bnd, keff)
                        if found is not None:
                            return found
                if not improved:
                    break
        return None

    def _confirm(self, x, z, o, bnd, keff):
        """A search candidate becomes a witness only if the real model flags it and the operation model admits it."""
        lp, zi, ov = self._ops_lp(x, bnd, keff)
        vec = np.zeros(len(lp.lb))
        k = o["k"]
        removed = x["IN_BYTES"] - z["IN_BYTES"]
        sB = min(max(removed, ops.DUMMY_MIN * k), ops.MTU * k)
        vec[ov["k"]], vec[ov["d"]], vec[ov["u"]], vec[ov["v"]] = k, o["d"], o["u"], o["v"]
        vec[ov["ptot"]], vec[ov["sB"]] = removed - sB, sB
        for f, j in zi.items():
            vec[j] = z[f]
        if not self._ops_ok(lp, vec):
            return None
        zv = self._vec(z)
        m = float(self.ens.margin(zv[None, :])[0])
        if m < self.m_thr:
            return None
        changed = {f: (x[f], float(zv[i])) for i, f in enumerate(self.F) if float(zv[i]) != np.float32(x[f])}
        return {"witness_margin": m - self.m_thr, "witness": zv, "changed": changed,
                "ops": {"dummy_pkts": float(k), "padding_bytes": float(removed - sB), "dummy_bytes": float(sB),
                        "any_operation": float(o["u"]), "timing_changed": float(o["v"])}}

    # ---------------------------------------------------------------- certificate
    def check(self, xrow, time_limit: float = 1.0) -> dict:
        t0 = time.perf_counter()
        x = {f: float(np.float32(v)) for f, v in zip(self.F, xrow)}
        bad = out_of_model(x)
        if bad:
            return {"status": "UNRESOLVED", "stage": "out_of_model", "detail": ",".join(bad),
                    "seconds": time.perf_counter() - t0}
        lo, hi, b, keff = self._box(x)
        reach = [self._reach(tr, lo, hi) for tr in self.ens.trees]
        const = self.ens.base
        ub = self.ens.base
        active = []
        for ti, (leaves, internal) in enumerate(reach):
            vals = self.ens.trees[ti][3][leaves].astype(np.float64)
            if len(leaves) == 1:
                const += vals[0]
            else:
                active.append(ti)
            ub += vals.max()
        if ub < self.m_thr - FEAS_EPS:
            return {"status": "CERTIFIED", "stage": "box", "bound": ub - self.m_thr, "active_trees": 0,
                    "seconds": time.perf_counter() - t0}
        found = self._search(x, b, keff, t0 + self.search_share * time_limit)
        if found is not None:
            return {"status": "WITNESS", "stage": "search", "active_trees": len(active), **found,
                    "seconds": time.perf_counter() - t0}
        deadline = t0 + time_limit
        if deadline - time.perf_counter() <= 0.02:
            return {"status": "UNRESOLVED", "stage": "time_limit", "active_trees": len(active),
                    "seconds": time.perf_counter() - t0}
        out = self._milp(x, lo, hi, b, keff, reach, active, const, deadline, t0)
        if out["seconds"] > time_limit + LATE_GRACE and out["status"] != "UNRESOLVED":
            # an answer that arrives after the per-flow limit is recorded but not counted
            out = {**{k_: v_ for k_, v_ in out.items() if k_ not in ("witness", "changed", "ops")},
                   "status": "UNRESOLVED", "stage": "over_time", "late_status": out["status"]}
        return out

    def _ops_lp(self, x, bnd, keff):
        """Variables and rows of the operation model (shared by the solver and the witness check)."""
        P, T = self.budget["pad_bytes"], self.budget["delay_ms"]
        lp = _LP()
        k = lp.var(0, keff, integer=True)
        d = lp.var(0, 1 if keff else 0, integer=True)
        # u: some operation is used; v: timing changes (delay or dummy packets). They can only be 1 when the
        # budget allows such an operation for this flow.
        u = lp.var(0, 1 if (P > 0 or T > 0 or keff > 0) else 0, integer=True)
        v = lp.var(0, 1 if (T > 0 or keff > 0) else 0, integer=True)
        ptot = lp.var(0, P * x["IN_PKTS"])
        sB = lp.var(0, ops.MTU * keff)
        R = lp.row
        R({k: 1, d: -1}, 0, np.inf, "ops")
        R({k: 1, d: -keff}, -np.inf, 0, "ops")
        R({sB: 1, k: -ops.DUMMY_MIN}, 0, np.inf, "ops")
        R({sB: 1, k: -ops.MTU}, -np.inf, 0, "ops")
        R({ptot: 1, k: P}, -np.inf, P * x["IN_PKTS"], "ops")
        R({ptot: 1, u: -P * x["IN_PKTS"]}, -np.inf, 0, "ops")
        R({u: 1, v: -1}, 0, np.inf, "ops")
        R({v: 1, d: -1}, 0, np.inf, "ops")
        z = {f: lp.var(*bnd[f]) for f in bnd}
        R({z["IN_PKTS"]: 1, k: 1}, x["IN_PKTS"], x["IN_PKTS"], "ops")
        R({z["IN_BYTES"]: 1, ptot: 1, sB: 1}, x["IN_BYTES"], x["IN_BYTES"], "ops")
        rt = x["RETRANSMITTED_IN_PKTS"]
        R({z["RETRANSMITTED_IN_BYTES"]: 1, u: P * rt}, x["RETRANSMITTED_IN_BYTES"], np.inf, "ops")
        R({z["RETRANSMITTED_IN_BYTES"]: 1, z["IN_BYTES"]: -1}, -np.inf, 1.0, "ops")
        for f in ("FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT"):
            R({z[f]: 1, v: T}, x[f], np.inf, "ops")
        for f in ("DURATION_IN", "DURATION_OUT"):
            R({z[f]: 1, z["FLOW_DURATION_MILLISECONDS"]: -1}, -np.inf, 1.0, "ops")
        R({z["LONGEST_FLOW_PKT"]: 1, u: P, d: ops.MTU}, x["LONGEST_FLOW_PKT"], np.inf, "ops")
        R({z["MAX_IP_PKT_LEN"]: 1, z["LONGEST_FLOW_PKT"]: -1}, 0, 0, "ops")
        for f in ("SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN"):
            R({z[f]: 1, u: P}, x[f], np.inf, "ops")
            R({z[f]: 1, d: -ops.MTU}, -np.inf, x[f], "ops")
        R({z["MIN_IP_PKT_LEN"]: 1, z["SHORTEST_FLOW_PKT"]: -1}, -np.inf, 1.0, "ops")
        C = [sum(x[h] for h in ops.HIST[j:]) for j in range(5)]
        for j in range(5):
            R({z[h]: 1 for h in ops.HIST[j:]}, -np.inf, C[j], "ops")
        R({**{z[h]: 1 for h in ops.HIST}, k: 1}, C[0], np.inf, "ops")
        R({**{z[h]: 1 for h in ops.HIST}, k: 1}, x["IN_PKTS"] + x["OUT_PKTS"], np.inf, "ops")
        if P < ops.HIST_MIN_WIDTH:
            for j in range(1, 5):
                R({**{z[h]: 1 for h in ops.HIST[j - 1:]}, k: 1}, C[j], np.inf, "ops")
        for h in ops.HIST:
            R({z[h]: 1, u: -C[0]}, -np.inf, x[h], "ops")
            R({z[h]: 1, u: C[0]}, x[h], np.inf, "ops")
        for f in ops.SECOND_BYTES:
            M = bnd[f][1]
            R({z[f]: 1, u: -M}, -np.inf, x[f], "ops")
            R({z[f]: 1, u: M}, x[f], np.inf, "ops")
        for f in ops.IAT:
            M = bnd[f][1]
            R({z[f]: 1, v: -M}, -np.inf, x[f], "ops")
            R({z[f]: 1, v: M}, x[f], np.inf, "ops")
        R({z["CLIENT_TCP_FLAGS"]: 1, d: x["CLIENT_TCP_FLAGS"]}, x["CLIENT_TCP_FLAGS"], np.inf, "ops")
        R({z["TCP_FLAGS"]: 1, z["CLIENT_TCP_FLAGS"]: -1}, 0, np.inf, "ops")
        R({z["TCP_FLAGS"]: 1, d: x["TCP_FLAGS"] - x["SERVER_TCP_FLAGS"]}, x["TCP_FLAGS"], np.inf, "ops")
        R({z["TCP_WIN_MAX_IN"]: 1, d: x["TCP_WIN_MAX_IN"]}, x["TCP_WIN_MAX_IN"], np.inf, "ops")
        return lp, z, {"k": k, "d": d, "u": u, "v": v, "ptot": ptot, "sB": sB}

    def _ops_ok(self, lp, vec):
        """True when a full variable vector satisfies every bound and every operation-model row."""
        lb, ub = np.array(lp.lb), np.array(lp.ub)
        if np.any(vec < lb - 1e-6 * (1 + np.abs(lb))) or np.any(vec > ub + 1e-6 * (1 + np.abs(ub))):
            return False
        for co, a, b_, tag in lp.rows:
            if tag != "ops":
                continue
            val = sum(c * vec[j] for j, c in co.items())
            if val < a - 1e-6 * (1 + abs(a)) or val > b_ + 1e-6 * (1 + abs(b_)):
                return False
        return True

    def _milp(self, x, lo, hi, bnd, keff, reach, active, const, deadline, t0):
        lp, z, ov = self._ops_lp(x, bnd, keff)
        u, v = ov["u"], ov["v"]
        R = lp.row
        # split indicators: bvar[(feature, threshold)] = 1 when the preimage value is below the threshold
        bvar = {}
        for ti in active:
            left_, right_, feat, cond = self.ens.trees[ti]
            for n in reach[ti][1]:
                key = (int(feat[n]), float(cond[n]))
                if key not in bvar:
                    bvar[key] = lp.var(0, 1, integer=True)
        by_feat = {}
        for (fj, t), var in bvar.items():
            by_feat.setdefault(fj, []).append((t, var))
        dlo, dhi = bnd["FLOW_DURATION_MILLISECONDS"]
        zd = z["FLOW_DURATION_MILLISECONDS"]
        for fj, lst in by_feat.items():
            lst.sort()
            name = self.F[fj]
            for (t1, v1), (t2, v2) in zip(lst, lst[1:]):
                R({v1: 1, v2: -1}, -np.inf, 0, "tree")
            for t, bv in lst:
                if name in z:
                    zv = z[name]
                    l_, h_ = bnd[name]
                    M = (h_ - l_) + abs(t) + 2.0
                    if name in self.integer:
                        c = float(np.ceil(t))
                        R({zv: 1, bv: M}, -np.inf, c - 1 + M, "tree")
                        R({zv: 1, bv: M}, c, np.inf, "tree")
                    else:
                        R({zv: 1, bv: M}, -np.inf, t + M, "tree")
                        R({zv: 1, bv: M}, t, np.inf, "tree")
                elif name == "SRC_TO_DST_AVG_THROUGHPUT":
                    zb = z["IN_BYTES"]
                    M = 8000.0 * bnd["IN_BYTES"][1] * (1 + TOL_BR) + (abs(t) + 1) * (dhi + 1) + 10
                    R({zb: 8000.0 * (1 - TOL_BR), zd: -(t + 1), bv: M}, -np.inf, (t + 1) + M, "tree")
                    R({zd: (t - 1), zb: -8000.0 * (1 + TOL_BR), bv: -M}, -np.inf, 0.0, "tree")
                    bx = 1.0 if x[name] < t else 0.0
                    R({bv: 1, u: -1}, -np.inf, bx, "tree")
                    R({bv: 1, u: 1}, bx, np.inf, "tree")
                elif name == "DST_TO_SRC_AVG_THROUGHPUT":
                    ob = x["OUT_BYTES"]
                    M = 8000.0 * ob * (1 + TOL_BR) + (abs(t) + 1) * (dhi + 1) + 10
                    R({zd: -(t + 1), bv: M}, -np.inf, (t + 1) - 8000.0 * ob * (1 - TOL_BR) + M, "tree")
                    R({zd: (t - 1), bv: -M}, -np.inf, 8000.0 * ob * (1 + TOL_BR), "tree")
                    bx = 1.0 if x[name] < t else 0.0
                    R({bv: 1, v: -1}, -np.inf, bx, "tree")
                    R({bv: 1, v: 1}, bx, np.inf, "tree")
                else:
                    raise RuntimeError(f"split on unchanged feature {name} left open")

        obj = {}
        for ti in active:
            left_, right_, feat, cond = self.ens.trees[ti]
            leaves, internal = reach[ti]
            lv = {n: lp.var(0, 1, integer=True) for n in leaves}
            R({lv[n]: 1 for n in leaves}, 1, 1, "tree")
            for n in leaves:
                obj[lv[n]] = float(cond[n])
            under = {}

            def collect(n):
                if n in under:
                    return under[n]
                if left_[n] == -1:
                    under[n] = [n] if n in lv else []
                else:
                    under[n] = collect(left_[n]) + collect(right_[n])
                return under[n]
            for n in internal:
                bv = bvar[(int(feat[n]), float(cond[n]))]
                L, Rr = collect(left_[n]), collect(right_[n])
                if L:
                    R({**{lv[m]: 1 for m in L}, bv: -1}, -np.inf, 0, "tree")
                if Rr:
                    R({**{lv[m]: 1 for m in Rr}, bv: 1}, -np.inf, 1, "tree")
        need = self.m_thr - const - FEAS_EPS
        R(dict(obj), need, np.inf, "tree")

        nv = len(lp.lb)
        rows, cols, vals, rlo, rhi = [], [], [], [], []
        for i, (co, a, bb, tag) in enumerate(lp.rows):
            for j, c in co.items():
                rows.append(i)
                cols.append(j)
                vals.append(c)
            rlo.append(a)
            rhi.append(bb)
        A = csr_matrix((vals, (rows, cols)), shape=(len(lp.rows), nv))
        # A certificate is a yes/no question, so the solver looks for any preimage over the threshold and stops at
        # the first one (no objective). maximise=True instead searches for the strongest one.
        c = np.zeros(nv)
        if self.maximise:
            for j, val in obj.items():
                c[j] = -val
        res = milp(c, integrality=np.array(lp.integ), bounds=Bounds(lp.lb, lp.ub),
                   constraints=LinearConstraint(A, rlo, rhi),
                   options={"time_limit": float(max(0.01, deadline - time.perf_counter())), "disp": False,
                            "presolve": True})
        info = {"active_trees": len(active), "binaries": int(sum(lp.integ)), "rows": len(lp.rows)}
        if res.status == 2:
            return {"status": "CERTIFIED", "stage": "milp", **info, "seconds": time.perf_counter() - t0}
        if res.x is None:
            return {"status": "UNRESOLVED", "stage": "time_limit", **info, "seconds": time.perf_counter() - t0}
        wit = self._witness(x, res.x, z, bvar, u, v, A, rlo, rhi, lp)
        if wit is None:
            return {"status": "UNRESOLVED", "stage": "witness_failed_check", **info,
                    "seconds": time.perf_counter() - t0}
        return {"status": "WITNESS", "stage": "milp", **info, **wit, "seconds": time.perf_counter() - t0}

    def _witness(self, x, sol, z, bvar, u, v, A, rlo, rhi, lp):
        """Turn a solver point into a concrete preimage and keep it only if the real model flags it and it
        satisfies the operation model."""
        ops_rows = np.array([tag == "ops" for (_, _, _, tag) in lp.rows])
        for rounding in (True, False):
            s = sol.copy()
            zf = dict(x)
            for f, j in z.items():
                val = float(s[j])
                if rounding and f in self.integer:
                    val = float(np.clip(np.round(val), lp.lb[j], lp.ub[j]))
                s[j] = val
                zf[f] = val
            dur = zf["FLOW_DURATION_MILLISECONDS"]
            for name, byts, gate in (("SRC_TO_DST_AVG_THROUGHPUT", zf["IN_BYTES"], s[u]),
                                     ("DST_TO_SRC_AVG_THROUGHPUT", x["OUT_BYTES"], s[v])):
                if gate < 0.5:
                    zf[name] = x[name]
                    continue
                blo = 8000.0 * byts / (dur + 1)
                bhi = 8000.0 * byts / dur if dur > 0 else 8000.0 * byts
                if dur == 0:
                    blo = bhi = 8000.0 * byts
                fj = self.fi[name]
                lower, upper = blo, bhi
                for (bj, t), var in bvar.items():
                    if bj != fj:
                        continue
                    if s[var] > 0.5:
                        upper = min(upper, np.nextafter(np.float32(t), np.float32(-np.inf)))
                    else:
                        lower = max(lower, t)
                zf[name] = float((lower + upper) / 2) if lower <= upper else float(blo)
            viol = A[ops_rows] @ s
            lo_, hi_ = np.array(rlo)[ops_rows], np.array(rhi)[ops_rows]
            if np.any(viol < lo_ - 1e-5 * (1 + np.abs(lo_))) or np.any(viol > hi_ + 1e-5 * (1 + np.abs(hi_))):
                continue
            zv = np.array([zf[f] for f in self.F], dtype=np.float32)
            m = float(self.ens.margin(zv[None, :])[0])
            if m >= self.m_thr:
                changed = {f: (x[f], float(zv[i])) for i, f in enumerate(self.F) if float(zv[i]) != np.float32(x[f])}
                return {"witness_margin": m - self.m_thr, "witness": zv, "changed": changed,
                        "ops": {"dummy_pkts": float(s[0]), "padding_bytes": float(s[4]), "dummy_bytes": float(s[5]),
                                "any_operation": float(s[u]), "timing_changed": float(s[v])}}
        return None


# ------------------------------------------------------------------ batch runner (parallel, resumable)
_W = {}


def _init_worker(verifier, time_limit):
    _W["v"], _W["t"] = verifier, float(time_limit)


def _run_chunk(chunk):
    ids, rows = chunk
    out = []
    for i, row in zip(ids, rows):
        r = _W["v"].check(row, time_limit=_W["t"])
        changed = r.get("changed") or {}
        out.append({"row_idx": int(i), "status": r["status"], "stage": r.get("stage", ""),
                    "seconds": float(r["seconds"]), "active_trees": int(r.get("active_trees", 0)),
                    "detail": r.get("detail", r.get("late_status", "")),
                    "witness_margin": float(r.get("witness_margin", np.nan)),
                    "changed": json.dumps({f: [round(a, 4), round(b, 4)] for f, (a, b) in changed.items()}),
                    "n_changed": len(changed),
                    **{f"op_{k}": float(v) for k, v in (r.get("ops") or {}).items()}})
    return out


def run(verifier, X32: np.ndarray, row_ids, cache_path, time_limit=1.0, workers=None, chunk=25, log_every=500):
    """Check every row, in parallel, saving results to cache_path after each chunk so an interrupted run resumes
    where it stopped. Returns a DataFrame with one row per flow."""
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import pandas as pd
    cache_path = str(cache_path)
    done = pd.read_parquet(cache_path) if os.path.exists(cache_path) else pd.DataFrame()
    have = set(done["row_idx"]) if len(done) else set()
    todo = [(i, r) for i, r in zip(row_ids, X32) if int(i) not in have]
    chunks = [([i for i, _ in todo[s:s + chunk]], [r for _, r in todo[s:s + chunk]]) for s in range(0, len(todo), chunk)]
    frames = [done] if len(done) else []
    workers = workers or physical_cores()
    t0, n = time.time(), len(have)
    if chunks:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(verifier, time_limit)) as ex:
            futs = [ex.submit(_run_chunk, c) for c in chunks]
            for fu in as_completed(futs):
                frames.append(pd.DataFrame(fu.result()))
                n += len(frames[-1])
                pd.concat(frames, ignore_index=True).to_parquet(cache_path, index=False)
                if n % log_every < chunk:
                    print(f"  {n:,}/{len(row_ids):,} checked, {time.time() - t0:.0f}s", flush=True)
    res = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return res[res["row_idx"].isin(set(int(i) for i in row_ids))].reset_index(drop=True)


def physical_cores() -> int:
    """One worker per physical core: a per-flow time limit is only fair if each check has a whole core."""
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
    except Exception:
        n = None
    return max(1, int(n or 1))


def machine() -> dict:
    import os
    import platform
    cpu = platform.processor() or ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "physical_cores": physical_cores()}


def summarise(res, label: dict):
    n = len(res)
    row = {**label, "flows": n}
    for s_ in STATUS:
        row[f"{s_.lower()}_share"] = float((res["status"] == s_).mean()) if n else np.nan
        row[s_.lower()] = int((res["status"] == s_).sum())
    row["out_of_model"] = int((res["stage"] == "out_of_model").sum())
    row["over_time"] = int((res["stage"] == "over_time").sum())
    row["median_seconds"] = float(res["seconds"].median()) if n else np.nan
    row["p95_seconds"] = float(res["seconds"].quantile(0.95)) if n else np.nan
    return row
