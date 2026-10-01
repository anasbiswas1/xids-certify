"""Attacker operations on NetFlow v3 flows, as used by the certificate.

The attacker is the sending side of the flow and changes only its own packets: it pads them, adds dummy packets,
and delays them. Budgets are fixed here before any certificate result. Each detector feature is either unchanged
by these operations, changed exactly, changed within a range, or free (no audited formula ties it to the rest).
"""
from __future__ import annotations

import numpy as np

BUDGETS = {
    "primary": {"pad_bytes": 100, "dummy_pkts": 5, "delay_ms": 1000},
    "low": {"pad_bytes": 20, "dummy_pkts": 1, "delay_ms": 100},
    "high": {"pad_bytes": 400, "dummy_pkts": 20, "delay_ms": 5000},
}
DUMMY_MIN, MTU = 40, 1514
HIST = ["NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES",
        "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES"]
HIST_MIN_WIDTH = 128          # padding below this moves a packet up at most one bin
IAT = [f"{d}_IAT_{s}" for d in ("SRC_TO_DST", "DST_TO_SRC") for s in ("MIN", "MAX", "AVG", "STDDEV")]
SECOND_BYTES = ["SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES"]
THROUGHPUT = ["SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT"]
# cumulative size counts: packets at least as large as each bin's lower edge (the first one is every packet)
CUM = ["CUM_PKTS_ALL", "CUM_PKTS_OVER_128", "CUM_PKTS_OVER_256", "CUM_PKTS_OVER_512", "CUM_PKTS_OVER_1024"]
UNMODELLED = IAT + SECOND_BYTES
# preimage value <= observed value under every operation (the attacker can only push these up) ...
INCREASE_ONLY = ["IN_PKTS", "IN_BYTES", "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
                 "LONGEST_FLOW_PKT", "MAX_IP_PKT_LEN", "RETRANSMITTED_IN_BYTES", "TCP_WIN_MAX_IN",
                 "CLIENT_TCP_FLAGS", "TCP_FLAGS"] + CUM
# ... or >= observed value (delay can only lower the receiver-side throughput)
DECREASE_ONLY = ["DST_TO_SRC_AVG_THROUGHPUT"]
# can move either way: padding raises them, a small dummy packet lowers them; delay lowers, padding raises
BOTH_WAYS = ["SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "SRC_TO_DST_AVG_THROUGHPUT"]
UNCHANGED = ["PROTOCOL", "L7_PROTO", "OUT_BYTES", "OUT_PKTS", "SERVER_TCP_FLAGS", "RETRANSMITTED_IN_PKTS",
             "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS", "TCP_WIN_MAX_OUT", "ICMP_TYPE", "ICMP_IPV4_TYPE",
             "DNS_QUERY_TYPE", "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE"]
FEATURE_ROLES = {
    "unchanged": UNCHANGED,
    "exact": ["IN_PKTS", "IN_BYTES", "RETRANSMITTED_IN_BYTES"],
    "range": ["FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT", "LONGEST_FLOW_PKT", "MAX_IP_PKT_LEN",
              "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "CLIENT_TCP_FLAGS", "TCP_FLAGS", "TCP_WIN_MAX_IN"] + HIST,
    "cumulative_counts": CUM,
    "bracket": THROUGHPUT,
    "free_when_timing_changes": IAT,
    "free_when_any_operation": SECOND_BYTES,
}
OPERATIONS_CONFIG = {
    "version": "operations_v2",
    "fixed_on": "2026-09-30; v1 before any certificate result, v2 after notebook 03 and before any v2 result",
    "changes_from_v1": ["cumulative size counts and operation directions added",
                        "receiver-side throughput of the original is at least the observed value",
                        "any drop in smallest or longest packet must be paid for with padding bytes",
                        "simulation carries one exact duration per flow"],
    "attacker": "the sending side of the flow; changes only its own packets",
    "budgets": BUDGETS,
    "primary_budget": "primary (pass/fail); low and high are sensitivity only",
    "dummy_packet_size_bytes": [DUMMY_MIN, MTU],
    "feature_roles": FEATURE_ROLES,
    "assumptions": [
        "the receiver does not answer dummy packets, so OUT_* counts are unchanged",
        "padding, dummy packets and delay do not change the protocol, the application protocol or the receiver's fields",
        "dummy packets may carry any client TCP flags and any advertised window",
        "fields with no audited formula (per-second bytes, inter-arrival statistics) are free whenever the "
        "operations that could affect them are used",
    ],
    "direction": {"increase_only": INCREASE_ONLY, "decrease_only": DECREASE_ONLY, "both_ways": BOTH_WAYS},
    "audited_relations_used": ["duration_in_le_total", "duration_out_le_total", "hist_sum_ge_all_packets",
                               "longest_eq_max_ip_len", "longest_le_1514", "out_bytes_within_pkt_bounds",
                               "retrans_in_le_totals", "shortest_ge_min_ip_len", "tcp_flags_or",
                               "thr_in_bracket_floor_ms", "thr_out_bracket_floor_ms"],
    "audited_relations_not_used": {"in_bytes_le_pkts_x_longest": "product of two unknowns; leaving it out only "
                                   "widens the preimage", "duration_from_timestamps": "timestamps are not model inputs",
                                   "ttl_order": "TTL is not a detector input", "out_zero_consistency": "OUT_* unchanged",
                                   "retrans_out_le_totals": "OUT_* unchanged"},
}


def add_cumulative(frame):
    """Cumulative size counts from the five bins; works on a DataFrame or a dict of one flow."""
    for j, c in enumerate(CUM):
        frame[c] = sum(frame[h] for h in HIST[j:])
    return frame


def integer_features(X, features, limit=2 ** 24):
    """Features whose training values are all whole numbers below 2^24 (exact in float32)."""
    out = []
    for f in features:
        v = np.asarray(X[f], dtype=np.float64)
        if np.all(np.isfinite(v)) and np.all(v == np.round(v)) and np.abs(v).max() < limit:
            out.append(f)
    return out


def apply_operations(z: dict, budget: dict, rng, plan=None) -> dict:
    """Forward simulation on one flow's features, inside the same model the certificate reasons about.
    Used only to test the certificate: any flow produced here from a flagged flow must never be certified."""
    P, K, T = budget["pad_bytes"], budget["dummy_pkts"], budget["delay_ms"]
    x = dict(z)
    zin = int(z["IN_PKTS"])
    plan = plan or {}
    k = int(plan.get("k", rng.integers(0, K + 1)))
    p = int(plan.get("pad", rng.integers(0, P + 1)))
    delay = float(plan.get("delay", rng.uniform(0, T)))
    sizes = rng.integers(DUMMY_MIN, MTU + 1, size=k)
    timing = k > 0 or delay > 0
    used = timing or p > 0
    x["IN_PKTS"] = zin + k
    x["IN_BYTES"] = z["IN_BYTES"] + p * zin + float(sizes.sum())
    x["RETRANSMITTED_IN_BYTES"] = z["RETRANSMITTED_IN_BYTES"] + p * z["RETRANSMITTED_IN_PKTS"]
    src_long, src_short = rng.random() < 0.5, rng.random() < 0.5
    lng = min(z["LONGEST_FLOW_PKT"] + (p if src_long else 0.0), MTU)
    sht = z["SHORTEST_FLOW_PKT"] + (p if src_short else 0.0)
    mip = z["MIN_IP_PKT_LEN"] + (p if src_short else 0.0)
    if z["OUT_PKTS"] > 0:
        # the flow-wide smallest packet can never exceed the receiver's smallest, which is at most its average
        cap = np.floor(z["OUT_BYTES"] / z["OUT_PKTS"])
        sht, mip = min(sht, max(cap, z["SHORTEST_FLOW_PKT"])), min(mip, max(cap, z["MIN_IP_PKT_LEN"]))
    if k:
        lng, sht, mip = max(lng, sizes.max()), min(sht, sizes.min()), min(mip, sizes.min())
    x["LONGEST_FLOW_PKT"] = x["MAX_IP_PKT_LEN"] = float(np.floor(lng))
    x["SHORTEST_FLOW_PKT"] = float(np.floor(min(sht, x["LONGEST_FLOW_PKT"])))
    x["MIN_IP_PKT_LEN"] = float(np.floor(min(mip, x["SHORTEST_FLOW_PKT"])))
    bins = np.array([z[b] for b in HIST], dtype=np.float64)
    if p > 0 and P < HIST_MIN_WIDTH:
        moved = np.zeros(5)
        budget_pkts = zin
        for j in range(4):
            m = int(rng.integers(0, int(min(bins[j], budget_pkts)) + 1)) if bins[j] > 0 and budget_pkts > 0 else 0
            moved[j] -= m
            moved[j + 1] += m
            budget_pkts -= m
        bins = bins + moved
    elif p > 0:
        budget_pkts = zin
        for j in range(4):
            m = int(rng.integers(0, int(min(bins[j], budget_pkts)) + 1)) if bins[j] > 0 and budget_pkts > 0 else 0
            dest = int(rng.integers(j + 1, 5))
            bins[j] -= m
            bins[dest] += m
            budget_pkts -= m
    for s in sizes:
        bins[int(np.searchsorted([128, 256, 512, 1024], s, side="left"))] += 1
    for b, v in zip(HIST, bins):
        x[b] = float(v)
    # nProbe computes throughput from the exact duration and stores whole milliseconds (audited bracket), so the
    # simulation carries one exact duration: the original's, recovered from its throughput, plus the added delay
    zd = z["FLOW_DURATION_MILLISECONDS"]
    if zd > 0 and z["OUT_BYTES"] > 0 and z["DST_TO_SRC_AVG_THROUGHPUT"] > 0:
        exact = min(max(8000.0 * z["OUT_BYTES"] / z["DST_TO_SRC_AVG_THROUGHPUT"], zd), zd + 0.999)
    elif zd > 0 and z["IN_BYTES"] > 0 and z["SRC_TO_DST_AVG_THROUGHPUT"] > 0:
        exact = min(max(8000.0 * z["IN_BYTES"] / z["SRC_TO_DST_AVG_THROUGHPUT"], zd), zd + 0.999)
    else:
        exact = zd + (rng.uniform(0, 0.999) if zd > 0 else 0.0)
    exact_x = exact
    if timing:
        add = rng.uniform(0, delay) if delay > 0 else 0.0
        exact_x = exact + add
        x["FLOW_DURATION_MILLISECONDS"] = float(np.floor(exact_x)) if zd > 0 or add > 0 else 0.0
        for d in ("DURATION_IN", "DURATION_OUT"):
            x[d] = float(min(np.floor(z[d] + rng.uniform(0, add)), x["FLOW_DURATION_MILLISECONDS"]))
        for f in IAT:
            x[f] = float(np.floor(max(0.0, z[f] * rng.uniform(0, 3) + rng.uniform(0, T))))
    if used:
        for f in SECOND_BYTES:
            x[f] = float(max(0.0, z[f] * rng.uniform(0, 3)))
    dur = x["FLOW_DURATION_MILLISECONDS"]
    for f, byts in (("SRC_TO_DST_AVG_THROUGHPUT", x["IN_BYTES"]), ("DST_TO_SRC_AVG_THROUGHPUT", z["OUT_BYTES"])):
        if dur == 0:
            x[f] = float(8000.0 * byts)
        elif f == "DST_TO_SRC_AVG_THROUGHPUT":
            # receiver bytes are unchanged and the flow can only get longer, so this can only fall
            x[f] = z[f] if not timing else float(min(z[f], 8000.0 * byts / max(exact_x, dur)))
        else:
            x[f] = float(8000.0 * byts / max(exact_x, dur))
    if k:
        x["CLIENT_TCP_FLAGS"] = float(int(z["CLIENT_TCP_FLAGS"]) | int(rng.choice([0, 8, 16, 24])))
        x["TCP_FLAGS"] = float(int(x["CLIENT_TCP_FLAGS"]) | int(z["SERVER_TCP_FLAGS"]))
        x["TCP_WIN_MAX_IN"] = float(max(z["TCP_WIN_MAX_IN"], rng.integers(0, 65536)))
    for f in x:
        x[f] = float(np.float32(x[f]))
    return x
