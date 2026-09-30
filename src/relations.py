"""Feature-relation audit for NetFlow v3.

Checks, on the data itself, which relations between NetFlow fields hold. The attacker operation model built later
may only use relations this audit shows to be exact (or bounded) on both datasets; names alone are not trusted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HIST_BINS = ["NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES",
             "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES"]
ABS_TOL = 1.0      # counts, bytes, milliseconds
REL_TOL = 0.01     # ratio relations


def _share(mask_ok, applicable):
    n = int(applicable.sum())
    return n, (float(mask_ok[applicable].mean()) if n else np.nan)


def _q(r):
    r = r[np.isfinite(r)]
    if not len(r):
        return {"q01": np.nan, "q50": np.nan, "q99": np.nan}
    return dict(zip(["q01", "q50", "q99"], np.quantile(r, [0.01, 0.5, 0.99]).round(6)))


def audit(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """One row per candidate relation: rows it applies to, share satisfied, and ratio quantiles where relevant."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return _audit(df, dataset)


def _audit(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    g = {c: df[c].to_numpy(dtype=np.float64) for c in df.columns if df[c].dtype.kind in "fiu"}
    rows = []

    def exact(name, formula, lhs, rhs, applicable=None, tol=ABS_TOL):
        app = np.ones(len(lhs), bool) if applicable is None else applicable
        n, s = _share(np.abs(lhs - rhs) <= tol, app)
        rows.append({"dataset": dataset, "relation": name, "formula": formula, "kind": "equality",
                     "rows_applicable": n, "share_satisfied": s, **_q((lhs - rhs)[app])})

    def bound(name, formula, ok, applicable=None):
        app = np.ones(len(ok), bool) if applicable is None else applicable
        n, s = _share(ok, app)
        rows.append({"dataset": dataset, "relation": name, "formula": formula, "kind": "inequality",
                     "rows_applicable": n, "share_satisfied": s, "q01": np.nan, "q50": np.nan, "q99": np.nan})

    def ratio(name, formula, num, den, applicable):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = num / den
        n, s = _share(np.abs(r - 1.0) <= REL_TOL, applicable)
        rows.append({"dataset": dataset, "relation": name, "formula": formula, "kind": "ratio = 1",
                     "rows_applicable": n, "share_satisfied": s, **_q(r[applicable])})

    dur = g["FLOW_DURATION_MILLISECONDS"]
    inb, outb, inp, outp = g["IN_BYTES"], g["OUT_BYTES"], g["IN_PKTS"], g["OUT_PKTS"]
    din, dout = g["DURATION_IN"], g["DURATION_OUT"]
    lo, hi = g["SHORTEST_FLOW_PKT"], g["LONGEST_FLOW_PKT"]
    pos = dur > 0

    if "flow_start_ms" in g and "flow_end_ms" in g:
        exact("duration_from_timestamps", "FLOW_DURATION = flow_end_ms - flow_start_ms", dur,
              g["flow_end_ms"] - g["flow_start_ms"])
    bound("duration_in_le_total", "DURATION_IN <= FLOW_DURATION", din <= dur + ABS_TOL)
    bound("duration_out_le_total", "DURATION_OUT <= FLOW_DURATION", dout <= dur + ABS_TOL)
    hist = sum(g[b] for b in HIST_BINS)
    exact("hist_sum_all_packets", "sum(NUM_PKTS_* bins) = IN_PKTS + OUT_PKTS", hist, inp + outp)
    exact("hist_sum_in_packets", "sum(NUM_PKTS_* bins) = IN_PKTS", hist, inp)
    exact("shortest_eq_min_ip_len", "SHORTEST_FLOW_PKT = MIN_IP_PKT_LEN", lo, g["MIN_IP_PKT_LEN"])
    exact("longest_eq_max_ip_len", "LONGEST_FLOW_PKT = MAX_IP_PKT_LEN", hi, g["MAX_IP_PKT_LEN"])
    bound("in_bytes_within_pkt_bounds", "IN_PKTS*SHORTEST <= IN_BYTES <= IN_PKTS*LONGEST",
          (inp * lo <= inb + ABS_TOL) & (inb <= inp * hi + ABS_TOL), inp > 0)
    bound("out_bytes_within_pkt_bounds", "OUT_PKTS*SHORTEST <= OUT_BYTES <= OUT_PKTS*LONGEST",
          (outp * lo <= outb + ABS_TOL) & (outb <= outp * hi + ABS_TOL), outp > 0)
    bound("longest_le_1514", "LONGEST_FLOW_PKT <= 1514", hi <= 1514)
    bound("out_zero_consistency", "OUT_PKTS = 0 iff OUT_BYTES = 0", (outp == 0) == (outb == 0))
    ratio("thr_in_over_total_duration", "SRC_TO_DST_AVG_THROUGHPUT / (IN_BYTES*8000/FLOW_DURATION)",
          g["SRC_TO_DST_AVG_THROUGHPUT"], inb * 8000.0 / dur, pos)
    ratio("thr_in_over_duration_in", "SRC_TO_DST_AVG_THROUGHPUT / (IN_BYTES*8000/DURATION_IN)",
          g["SRC_TO_DST_AVG_THROUGHPUT"], inb * 8000.0 / din, din > 0)
    ratio("thr_out_over_total_duration", "DST_TO_SRC_AVG_THROUGHPUT / (OUT_BYTES*8000/FLOW_DURATION)",
          g["DST_TO_SRC_AVG_THROUGHPUT"], outb * 8000.0 / dur, pos & (outb > 0))
    ratio("thr_out_over_duration_out", "DST_TO_SRC_AVG_THROUGHPUT / (OUT_BYTES*8000/DURATION_OUT)",
          g["DST_TO_SRC_AVG_THROUGHPUT"], outb * 8000.0 / dout, (dout > 0) & (outb > 0))
    ratio("sec_bytes_in_over_rate", "SRC_TO_DST_SECOND_BYTES / (IN_BYTES*1000/FLOW_DURATION)",
          g["SRC_TO_DST_SECOND_BYTES"], inb * 1000.0 / dur, pos)
    ratio("sec_bytes_in_over_bytes", "SRC_TO_DST_SECOND_BYTES / IN_BYTES",
          g["SRC_TO_DST_SECOND_BYTES"], inb, inb > 0)
    ratio("sec_bytes_in_over_bytes_per_pkt", "SRC_TO_DST_SECOND_BYTES / (IN_BYTES/IN_PKTS)",
          g["SRC_TO_DST_SECOND_BYTES"], inb / inp, inp > 0)
    ratio("sec_bytes_out_over_rate", "DST_TO_SRC_SECOND_BYTES / (OUT_BYTES*1000/FLOW_DURATION)",
          g["DST_TO_SRC_SECOND_BYTES"], outb * 1000.0 / dur, pos & (outb > 0))
    ratio("sec_bytes_out_over_bytes_per_pkt", "DST_TO_SRC_SECOND_BYTES / (OUT_BYTES/OUT_PKTS)",
          g["DST_TO_SRC_SECOND_BYTES"], outb / outp, outp > 0)
    # brackets: a field computed from the exact duration lies between the values implied by the whole-millisecond
    # duration stored in the file (floor) and that duration plus one millisecond
    tol = 1e-6
    for nm, fld, byts, app in (("in", "SRC_TO_DST_AVG_THROUGHPUT", inb, pos),
                               ("out", "DST_TO_SRC_AVG_THROUGHPUT", outb, pos)):
        v = g[fld]
        lo_b, hi_b = byts * 8000.0 / (dur + 1.0), byts * 8000.0 / dur
        bound(f"thr_{nm}_bracket_floor_ms", f"{fld} in [BYTES*8000/(FLOW_DURATION+1), BYTES*8000/FLOW_DURATION]",
              (v >= lo_b * (1 - tol) - 1) & (v <= hi_b * (1 + tol) + 1), app)
    exact("thr_in_zero_duration_equals_bytes_x8000", "SRC_TO_DST_AVG_THROUGHPUT = IN_BYTES*8000 when FLOW_DURATION = 0",
          g["SRC_TO_DST_AVG_THROUGHPUT"], inb * 8000.0, ~pos)
    bound("hist_sum_ge_all_packets", "sum(NUM_PKTS_* bins) >= IN_PKTS + OUT_PKTS", hist >= inp + outp)
    bound("hist_sum_le_all_packets", "sum(NUM_PKTS_* bins) <= IN_PKTS + OUT_PKTS", hist <= inp + outp)
    bound("sec_bytes_in_le_in_bytes", "SRC_TO_DST_SECOND_BYTES <= IN_BYTES", g["SRC_TO_DST_SECOND_BYTES"] <= inb + ABS_TOL)
    bound("sec_bytes_out_le_out_bytes", "DST_TO_SRC_SECOND_BYTES <= OUT_BYTES",
          g["DST_TO_SRC_SECOND_BYTES"] <= outb + ABS_TOL)
    bound("shortest_le_min_ip_len", "SHORTEST_FLOW_PKT <= MIN_IP_PKT_LEN", lo <= g["MIN_IP_PKT_LEN"] + ABS_TOL)
    bound("shortest_ge_min_ip_len", "SHORTEST_FLOW_PKT >= MIN_IP_PKT_LEN", lo >= g["MIN_IP_PKT_LEN"] - ABS_TOL)
    bound("in_bytes_ge_pkts_x_shortest", "IN_BYTES >= IN_PKTS*SHORTEST_FLOW_PKT", inp * lo <= inb + ABS_TOL, inp > 0)
    bound("in_bytes_le_pkts_x_longest", "IN_BYTES <= IN_PKTS*LONGEST_FLOW_PKT", inb <= inp * hi + ABS_TOL, inp > 0)
    for d, npk, dd in (("SRC_TO_DST", inp, din), ("DST_TO_SRC", outp, dout)):
        av = g[f"{d}_IAT_AVG"]
        bound(f"iat_avg_{d.lower()}_le_stream_duration_bracket",
              f"{d}_IAT_AVG in [(stream duration - packets)/(packets - 1), (stream duration + 1)/(packets - 1)]",
              (av >= (dd - npk) / np.maximum(npk - 1, 1) - ABS_TOL) & (av <= (dd + 1) / np.maximum(npk - 1, 1) + ABS_TOL),
              npk > 1)
    tf, cf, sf = (df[c].to_numpy(dtype=np.int64) for c in ("TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS"))
    exact("tcp_flags_or", "TCP_FLAGS = CLIENT_TCP_FLAGS | SERVER_TCP_FLAGS", tf.astype(float), (cf | sf).astype(float),
          tol=0.0)
    bound("ttl_order", "MIN_TTL <= MAX_TTL", g["MIN_TTL"] <= g["MAX_TTL"])
    for d, npk, dd in (("SRC_TO_DST", inp, din), ("DST_TO_SRC", outp, dout)):
        mn, av, mx = g[f"{d}_IAT_MIN"], g[f"{d}_IAT_AVG"], g[f"{d}_IAT_MAX"]
        bound(f"iat_order_{d.lower()}", f"{d}_IAT_MIN <= AVG <= MAX", (mn <= av + ABS_TOL) & (av <= mx + ABS_TOL),
              npk > 1)
        bound(f"iat_zero_when_single_packet_{d.lower()}", f"{d}_IAT_* = 0 when packets <= 1",
              (mn == 0) & (av == 0) & (mx == 0), npk <= 1)
        exact(f"iat_avg_{d.lower()}_from_stream_duration", f"{d}_IAT_AVG = stream duration / (packets - 1)",
              av, dd / np.maximum(npk - 1, 1), npk > 1)
        exact(f"iat_avg_{d.lower()}_from_flow_duration", f"{d}_IAT_AVG = FLOW_DURATION / (packets - 1)",
              av, dur / np.maximum(npk - 1, 1), npk > 1)
        bound(f"iat_max_le_duration_{d.lower()}", f"{d}_IAT_MAX <= FLOW_DURATION", mx <= dur + ABS_TOL)
    bound("retrans_in_le_totals", "RETRANSMITTED_IN_PKTS <= IN_PKTS and bytes <= IN_BYTES",
          (g["RETRANSMITTED_IN_PKTS"] <= inp) & (g["RETRANSMITTED_IN_BYTES"] <= inb + ABS_TOL))
    bound("retrans_out_le_totals", "RETRANSMITTED_OUT_PKTS <= OUT_PKTS and bytes <= OUT_BYTES",
          (g["RETRANSMITTED_OUT_PKTS"] <= outp) & (g["RETRANSMITTED_OUT_BYTES"] <= outb + ABS_TOL))
    return pd.DataFrame(rows)


def verdict(tables: pd.DataFrame, exact_share: float = 0.999) -> pd.DataFrame:
    """A relation is usable by the attacker model only if it holds on at least 99.9% of applicable rows in every
    dataset; anything weaker is reported but not used."""
    p = tables.pivot_table(index=["relation", "formula", "kind"], columns="dataset", values="share_satisfied")
    p["usable_in_operation_model"] = (p >= exact_share).all(axis=1)
    return p.reset_index()
