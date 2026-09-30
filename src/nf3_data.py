"""NetFlow v3 data stage for X-IDS-Certify.

Identifies the NF3 archives, checks them against the published schema and row counts, applies the binary label
policy, keeps zero-duration flows with their undefined rates set to 0, collapses exact feature duplicates (majority
label for groups at least 99% pure, quarantine otherwise), groups flows into conversation-minute clusters, assigns
clusters to four disjoint roles, and writes one parquet file per dataset outside the repository.
Two passes over each CSV keep memory bounded for the 27.5M-row ToN-IoT file.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ schema (Luay et al. 2025, arXiv 2503.04404, Table 1)
NF3_FEATURES = [
    "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO", "IN_BYTES",
    "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS", "TCP_FLAGS", "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS", "DURATION_IN", "DURATION_OUT", "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN", "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT", "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE", "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS", "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV", "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX", "DST_TO_SRC_IAT_AVG",
    "DST_TO_SRC_IAT_STDDEV",
]
IDENTIFIERS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT",
               "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"]
# A DNS transaction ID is a random 16-bit number chosen by the sender: no traffic meaning, a known shortcut risk.
EXCLUDED_FROM_MODEL = IDENTIFIERS + ["DNS_QUERY_ID"]
MODEL_FEATURES = [f for f in NF3_FEATURES if f not in EXCLUDED_FROM_MODEL]

PUBLISHED_ROWS = {"unsw": 2_365_424, "ton": 27_520_260, "cic2018": 20_115_529, "bot": 16_993_808}
DATASET_NAMES = {"unsw": "NF-UNSW-NB15-v3", "ton": "NF-ToN-IoT-v3", "cic2018": "NF-CSE-CIC-IDS2018-v3",
                 "bot": "NF-BoT-IoT-v3"}

ROLES = ["detector_train", "development", "calibration", "locked_test"]
ROLE_SHARES = {"detector_train": 0.50, "development": 0.15, "calibration": 0.15, "locked_test": 0.20}
SPLIT_SEED = 20260930
CLUSTER_WINDOW_MS = 60_000
CHUNK_ROWS = 1_000_000
# nProbe leaves the per-second byte rates empty when a flow lasts 0 ms; the rate is undefined, not unknown.
RATE_FEATURES = ["SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES"]
# A duplicate group whose copies disagree on Label keeps its majority label when at least this share agrees.
PURITY_MIN = 0.99

DATA_CONFIG = {
    "version": "nf3_v1",
    "source": "NetFlow v3 datasets, University of Queensland (Luay et al. 2025, arXiv 2503.04404)",
    "model_features": MODEL_FEATURES,
    "excluded_from_model": {"identifiers": IDENTIFIERS, "dns_query_id": "random transaction id"},
    "label_policy": "binary Label column; rows whose Label contradicts the Attack column are quarantined",
    "missing": "per-second byte rates of zero-duration flows are set to 0; any other row with a missing or "
               "non-finite model feature is dropped and counted",
    "duplicates": "exact model-feature duplicates collapse to one row with a multiplicity count; a group whose "
                  "rows disagree on Label keeps its majority label when at least 99% of its rows agree and is "
                  "quarantined otherwise; label purity and minority count are kept per row",
    "purity_min": 0.99,
    "clusters": "conversation-minute: (source address, destination address, destination port, protocol, "
                "60 s window of FLOW_START_MILLISECONDS); a cluster never crosses roles",
    "roles": ROLE_SHARES,
    "role_assignment": "clusters shuffled within each attack family (or Benign) and allocated by cumulative row share",
    "split_seed": SPLIT_SEED,
    "reference_population": "distinct flow records after duplicate collapse; multiplicity kept",
    "fitted_preprocessing": "none at this stage",
}


def norm_col(c: str) -> str:
    return re.sub(r"[^0-9A-Z]+", "_", str(c).strip().upper()).strip("_")


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def dataset_key(name: str):
    n = name.lower()
    if "unsw" in n:
        return "unsw"
    if "ton" in n:
        return "ton"
    if "bot" in n:
        return "bot"
    if "cic" in n or "2018" in n:
        return "cic2018"
    return None


# ------------------------------------------------------------------ archives
def find_archives(project: Path, repo: Path) -> pd.DataFrame:
    """Every zip or csv at the project root or under repo/data/raw, with the dataset each one holds."""
    cands = sorted(set(project.glob("*.zip")) | set(project.glob("*.csv")) |
                   set((repo / "data" / "raw").rglob("*.zip")) | set((repo / "data" / "raw").rglob("*.csv")))
    rows = []
    for p in cands:
        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as z:
                csvs = [i for i in z.infolist() if i.filename.lower().endswith(".csv") and "__macosx" not in i.filename.lower()]
            if not csvs:
                rows.append({"file": p.name, "member": None, "dataset": None, "bytes": p.stat().st_size,
                             "member_bytes": None, "path": str(p)})
                continue
            m = max(csvs, key=lambda i: i.file_size)
            rows.append({"file": p.name, "member": m.filename, "dataset": dataset_key(Path(m.filename).name),
                         "bytes": p.stat().st_size, "member_bytes": m.file_size, "path": str(p)})
        else:
            rows.append({"file": p.name, "member": None, "dataset": dataset_key(p.name), "bytes": p.stat().st_size,
                         "member_bytes": p.stat().st_size, "path": str(p)})
    return pd.DataFrame(rows)


def stage_local(archive: Path, work: Path) -> Path:
    """Copy one archive to the runtime disk (fast reads) and check every zip member's CRC."""
    work.mkdir(parents=True, exist_ok=True)
    local = work / archive.name
    if not local.exists() or local.stat().st_size != archive.stat().st_size:
        shutil.copy(archive, local)
    if local.suffix.lower() == ".zip":
        with zipfile.ZipFile(local) as z:
            bad = z.testzip()
        if bad is not None:
            raise RuntimeError(f"{archive.name}: corrupt member {bad}; re-download the file")
    return local


def _chunks(local: Path, member, chunksize=CHUNK_ROWS):
    if member:
        z = zipfile.ZipFile(local)
        fh = z.open(member)
    else:
        z, fh = None, open(local, "rb")
    try:
        for chunk in pd.read_csv(fh, chunksize=chunksize, low_memory=False):
            chunk.columns = [norm_col(c) for c in chunk.columns]
            yield chunk
    finally:
        fh.close()
        if z is not None:
            z.close()


def _label_cols(cols):
    lab = next((c for c in cols if c == "LABEL"), None)
    att = next((c for c in cols if c in ("ATTACK", "ATTACK_TYPE", "CLASS")), None)
    if lab is None or att is None:
        raise ValueError(f"Label/Attack columns not found; columns end with {cols[-5:]}")
    return lab, att


def _to_numeric(chunk):
    return chunk[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _model_matrix(chunk, X=None):
    """Model features as float64, with undefined rates of zero-duration flows set to 0 (same in both passes)."""
    X = _to_numeric(chunk) if X is None else X
    zero = (X["FLOW_DURATION_MILLISECONDS"] == 0).to_numpy()
    filled = np.zeros(len(X), dtype=bool)
    for c in RATE_FEATURES:
        m = zero & X[c].isna().to_numpy()
        if m.any():
            X.loc[m, c] = 0.0
            filled |= m
    return X, zero, filled


# ------------------------------------------------------------------ pass 1: scan
def scan(local: Path, member, dataset: str, chunksize=CHUNK_ROWS):
    """One pass: schema check, per-feature statistics, and the small per-row arrays the split needs."""
    parts = {k: [] for k in ("h", "y", "fam", "key", "ok", "zero", "filled")}
    fam_codes, stats, n = {}, None, 0
    schema_extra = schema_missing = None
    for ci, chunk in enumerate(_chunks(local, member, chunksize)):
        cols = list(chunk.columns)
        lab, att = _label_cols(cols)
        if ci == 0:
            present = set(cols) - {lab, att}
            schema_missing = [f for f in NF3_FEATURES if f not in present]
            schema_extra = sorted(present - set(NF3_FEATURES))
            if schema_missing:
                raise ValueError(f"{dataset}: columns missing from the published schema: {schema_missing}")
            stats = {f: {"min": np.inf, "max": -np.inf, "n_missing": 0, "n_zero": 0} for f in MODEL_FEATURES}
        X = _to_numeric(chunk)
        raw_missing = X.isna().sum()
        X, zero, filled = _model_matrix(chunk, X)
        ok = X.notna().all(axis=1).to_numpy()
        for f in MODEL_FEATURES:
            v = X[f].to_numpy(dtype=np.float64)
            s = stats[f]
            finite = v[~np.isnan(v)]
            if finite.size:
                s["min"] = min(s["min"], float(finite.min()))
                s["max"] = max(s["max"], float(finite.max()))
            s["n_missing"] += int(raw_missing[f])
            s["n_zero"] += int((finite == 0).sum())
        X32 = X.astype(np.float32)
        parts["h"].append(pd.util.hash_pandas_object(X32, index=False).to_numpy())
        y = pd.to_numeric(chunk[lab], errors="coerce").fillna(-1).astype(np.int8).to_numpy()
        fam = chunk[att].fillna("").astype(str).str.strip()
        codes = np.empty(len(fam), dtype=np.int16)
        for name in fam.unique():
            if name not in fam_codes:
                fam_codes[name] = len(fam_codes)
            codes[(fam == name).to_numpy()] = fam_codes[name]
        start = pd.to_numeric(chunk["FLOW_START_MILLISECONDS"], errors="coerce").fillna(-CLUSTER_WINDOW_MS)
        keydf = pd.DataFrame({
            "s": chunk["IPV4_SRC_ADDR"].fillna("").astype(str).to_numpy(),
            "d": chunk["IPV4_DST_ADDR"].fillna("").astype(str).to_numpy(),
            "p": pd.to_numeric(chunk["L4_DST_PORT"], errors="coerce").fillna(-1).astype(np.int64).to_numpy(),
            "t": pd.to_numeric(chunk["PROTOCOL"], errors="coerce").fillna(-1).astype(np.int64).to_numpy(),
            "m": (start // CLUSTER_WINDOW_MS).astype(np.int64).to_numpy(),
        })
        parts["key"].append(pd.util.hash_pandas_object(keydf, index=False).to_numpy())
        parts["y"].append(y)
        parts["fam"].append(codes)
        parts["ok"].append(ok)
        parts["zero"].append(zero)
        parts["filled"].append(filled)
        n += len(chunk)
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    families = {v: k for k, v in fam_codes.items()}
    schema = pd.DataFrame([{"dataset": dataset, "feature": f, **stats[f],
                            "constant": stats[f]["min"] == stats[f]["max"]} for f in MODEL_FEATURES])
    info = {"dataset": dataset, "rows_read": n, "rows_published": PUBLISHED_ROWS.get(dataset),
            "rows_match_published": n == PUBLISHED_ROWS.get(dataset), "extra_columns": schema_extra,
            "model_features": len(MODEL_FEATURES)}
    return arrays, families, schema, info


# ------------------------------------------------------------------ split decisions (in memory, small arrays)
def decide(arrays, families, dataset: str, seed: int = SPLIT_SEED):
    h, y, fam, key, ok = (arrays[k] for k in ("h", "y", "fam", "key", "ok"))
    n = len(h)
    benign_codes = {c for c, name in families.items() if name.lower() in ("benign", "normal")}
    is_benign_fam = np.isin(fam, list(benign_codes))
    label_conflict = ~np.isin(y, [0, 1]) | ((y == 0) & ~is_benign_fam) | ((y == 1) & is_benign_fam)
    audit = {"dataset": dataset, "rows_read": int(n), "rows_zero_duration": int(arrays["zero"].sum()),
             "rows_zero_duration_rate_set_0": int(arrays["filled"].sum()),
             "rows_missing_dropped": int((~ok).sum()),
             "rows_label_attack_conflict": int((label_conflict & ok).sum())}
    alive = ok & ~label_conflict
    idx = np.flatnonzero(alive)

    order = idx[np.argsort(h[idx], kind="stable")]
    hs = h[order]
    brk = np.flatnonzero(hs[1:] != hs[:-1]) + 1
    starts = np.r_[0, brk]
    ends = np.r_[brk, len(order)]
    ys = y[order].astype(np.int64)
    fs = fam[order]
    mult = (ends - starts).astype(np.int64)
    n_attack = np.add.reduceat(ys, starts)
    maj = (2 * n_attack > mult).astype(np.int64)          # ties go to benign
    n_major = np.where(maj == 1, n_attack, mult - n_attack)
    purity = n_major / mult
    mixed_label = (n_attack > 0) & (n_attack < mult)
    keep_run = purity >= PURITY_MIN
    # representative: the first row of the group (original file order) that carries the majority label
    pos = np.arange(len(order))
    cand = np.where(ys == np.repeat(maj, mult), pos, len(order))
    first_major = np.minimum.reduceat(cand, starts)
    run_mixed_fam = (np.minimum.reduceat(fs, starts) != np.maximum.reduceat(fs, starts)) & ~mixed_label
    keep = order[first_major[keep_run]]
    keep_mult = mult[keep_run]
    keep_minor = (mult - n_major)[keep_run]
    keep_purity = purity[keep_run]
    bins = [(0.5, 0.9), (0.9, 0.99), (0.99, 0.999), (0.999, 1.0)]
    audit.update({
        "duplicate_groups_label_mixed": int(mixed_label.sum()),
        "rows_in_label_mixed_groups": int(mult[mixed_label].sum()),
        "label_mixed_groups_kept_majority": int((mixed_label & keep_run).sum()),
        "minority_rows_absorbed": int((mult - n_major)[mixed_label & keep_run].sum()),
        "label_mixed_groups_quarantined": int((~keep_run).sum()),
        "rows_quarantined_duplicate_conflict": int(mult[~keep_run].sum()),
        "duplicate_groups_mixed_family_same_label": int(run_mixed_fam.sum()),
        "rows_after_collapse": int(len(keep)),
        "duplicate_groups_multiplicity_gt1": int((keep_mult > 1).sum()),
        "max_multiplicity": int(keep_mult.max()) if len(keep_mult) else 0,
    })
    for lo, hi in bins:
        sel = mixed_label & (purity >= lo) & (purity < hi)
        audit[f"mixed_groups_purity_{lo}_{hi}"] = int(sel.sum())
        audit[f"mixed_rows_purity_{lo}_{hi}"] = int(mult[sel].sum())

    srt = np.argsort(keep)
    keep, keep_mult, keep_minor, keep_purity = keep[srt], keep_mult[srt], keep_minor[srt], keep_purity[srt]
    cid, uniq = pd.factorize(key[keep], sort=True)
    cid = cid.astype(np.int64)
    n_cl = len(uniq)
    sizes = np.bincount(cid, minlength=n_cl)
    kf = fam[keep].astype(np.int64)
    nf = int(kf.max()) + 1 if len(kf) else 1
    pair, cnt = np.unique(cid * nf + kf, return_counts=True)
    pc, pf = pair // nf, pair % nf
    o = np.lexsort((pf, -cnt, pc))
    first = np.r_[True, pc[o][1:] != pc[o][:-1]]
    stratum = np.empty(n_cl, dtype=np.int64)
    stratum[pc[o][first]] = pf[o][first]

    rng = np.random.default_rng(seed)
    bounds = np.cumsum([ROLE_SHARES[r] for r in ROLES])
    role_of = np.full(n_cl, -1, dtype=np.int8)
    for s in np.unique(stratum):
        ids = np.flatnonzero(stratum == s)
        ids = ids[rng.permutation(len(ids))]
        sz = sizes[ids].astype(np.float64)
        mid = (np.cumsum(sz) - sz / 2) / sz.sum()
        role_of[ids] = np.minimum(np.searchsorted(bounds, mid, side="right"), len(ROLES) - 1)
    assert (role_of >= 0).all()
    role = role_of[cid]

    audit.update({"clusters": int(n_cl), "largest_cluster_rows": int(sizes.max()),
                  "largest_cluster_share": round(float(sizes.max() / len(keep)), 6)})
    for r_i, r in enumerate(ROLES):
        audit[f"share_{r}"] = round(float((role == r_i).mean()), 4)
    audit["attack_prevalence_after_collapse"] = round(float((y[keep] == 1).mean()), 4)
    hsh = hashlib.sha256()
    hsh.update(keep.astype(np.int64).tobytes())
    hsh.update(role.astype(np.int8).tobytes())
    audit["split_hash"] = hsh.hexdigest()

    plan = {"keep": keep, "mult": keep_mult, "minor": keep_minor, "purity": keep_purity, "cid": cid, "role": role}
    fam_name = np.array([families[c] for c in range(len(families))], dtype=object)
    by_role = (pd.DataFrame({"role": pd.Categorical.from_codes(role, ROLES), "y": y[keep]})
               .groupby(["role", "y"], observed=False).size().unstack(fill_value=0)
               .rename(columns={0: "benign", 1: "attack"}).reindex(ROLES).reset_index())
    for c in ("benign", "attack"):
        if c not in by_role:
            by_role[c] = 0
    by_role["rows"] = by_role["benign"] + by_role["attack"]
    by_role["attack_share"] = (by_role["attack"] / by_role["rows"]).round(4)
    by_role["target_share"] = by_role["role"].map(ROLE_SHARES)
    by_role["achieved_share"] = (by_role["rows"] / by_role["rows"].sum()).round(4)
    by_role.insert(0, "dataset", dataset)
    pr, pcnt = np.unique(fam[keep].astype(np.int64) * len(ROLES) + role.astype(np.int64), return_counts=True)
    by_family = (pd.DataFrame({"family": fam_name[pr // len(ROLES)], "role": np.array(ROLES)[pr % len(ROLES)],
                               "rows": pcnt})
                 .pivot_table(index="family", columns="role", values="rows", aggfunc="sum", fill_value=0)
                 .reindex(columns=ROLES, fill_value=0).reset_index())
    by_family.columns.name = None
    by_family.insert(0, "dataset", dataset)
    by_family["total"] = by_family[ROLES].sum(axis=1)
    by_family = by_family.sort_values("total", ascending=False).reset_index(drop=True)
    code4 = fam.astype(np.int64) * 4 + (y.astype(np.int64) + 1)
    lp, lcnt = np.unique(code4, return_counts=True)
    zp, zcnt = np.unique(code4[arrays["zero"]], return_counts=True)
    labels = pd.DataFrame({"family": fam_name[lp // 4], "y": (lp % 4 - 1).astype(np.int8), "rows_read": lcnt,
                           "zero_duration_rows": pd.Series(zcnt, index=zp).reindex(lp, fill_value=0).to_numpy()})
    labels.insert(0, "dataset", dataset)
    return plan, audit, by_role, by_family, labels


# ------------------------------------------------------------------ pass 2: write kept rows
def write_parquet(local: Path, member, plan, families, dataset: str, out_dir: Path, chunksize=CHUNK_ROWS) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path("/tmp") / f"{dataset}_flows.parquet"
    if tmp.exists():
        tmp.unlink()
    keep, mult, cid, role = plan["keep"], plan["mult"], plan["cid"], plan["role"]
    minor, purity = plan["minor"], plan["purity"]
    writer, offset, written = None, 0, 0
    for chunk in _chunks(local, member, chunksize):
        lab, att = _label_cols(list(chunk.columns))
        lo, hi = offset, offset + len(chunk)
        a, b = np.searchsorted(keep, lo), np.searchsorted(keep, hi)
        sel = keep[a:b] - lo
        if len(sel):
            c = chunk.iloc[sel]
            X = _model_matrix(c)[0].astype(np.float32).reset_index(drop=True)
            meta = pd.DataFrame({
                "row_idx": keep[a:b].astype(np.int64),
                "src_ip": c["IPV4_SRC_ADDR"].fillna("").astype(str).to_numpy(),
                "dst_ip": c["IPV4_DST_ADDR"].fillna("").astype(str).to_numpy(),
                "src_port": pd.to_numeric(c["L4_SRC_PORT"], errors="coerce").fillna(-1).astype(np.int64).to_numpy(),
                "dst_port": pd.to_numeric(c["L4_DST_PORT"], errors="coerce").fillna(-1).astype(np.int64).to_numpy(),
                "flow_start_ms": pd.to_numeric(c["FLOW_START_MILLISECONDS"], errors="coerce").fillna(-1).astype(np.int64).to_numpy(),
                "flow_end_ms": pd.to_numeric(c["FLOW_END_MILLISECONDS"], errors="coerce").fillna(-1).astype(np.int64).to_numpy(),
                "family": c[att].fillna("").astype(str).str.strip().to_numpy(),
                "y": pd.to_numeric(c[lab], errors="coerce").astype(np.int8).to_numpy(),
                "multiplicity": mult[a:b].astype(np.int64),
                "minority_rows": minor[a:b].astype(np.int64),
                "label_purity": purity[a:b].astype(np.float32),
                "cluster_id": cid[a:b].astype(np.int64),
                "role": np.array(ROLES, dtype=object)[role[a:b]],
            })
            table = pa.Table.from_pandas(pd.concat([meta, X], axis=1), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
            written += len(sel)
        offset = hi
    if writer is not None:
        writer.close()
    if written != len(keep):
        raise RuntimeError(f"{dataset}: wrote {written} rows, expected {len(keep)}")
    target = out_dir / "flows.parquet"
    shutil.copy(tmp, target)
    tmp.unlink()
    (out_dir / "features.json").write_text(json.dumps({"model_features": MODEL_FEATURES}, indent=2) + "\n")
    return target


def load_split(project: Path, dataset: str, roles=None, columns=None):
    """Read one dataset's parquet back, optionally only some roles."""
    import pyarrow.parquet as pq
    path = project / "data" / "nf3_v1" / dataset / "flows.parquet"
    filters = [("role", "in", list(roles))] if roles else None
    return pq.read_table(path, columns=columns, filters=filters).to_pandas()


def write_config(repo: Path) -> str:
    out = repo / "configs"
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(DATA_CONFIG, indent=2, sort_keys=True)
    (out / "data_nf3_v1.json").write_text(text + "\n")
    return hashlib.sha256(text.encode()).hexdigest()
