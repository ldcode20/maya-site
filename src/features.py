"""Phase 2 feature engineering for the maya project.

Reads data/transactions.csv, data/graph_edges.csv and data/labels.csv and builds
strictly past-only, time-aware fraud features for every transaction:

    n_t   distinct new counterparties in the last 7 days
    e_t   amount climb = current / max first-payment amount to new payees (14d)
    g_t   stranger-credit flag (unknown-sender credit to the account in 7d)
    z_t   unusual amount = log(amount) - median of past log amounts
    pi_t  probing counter = payments <= 10 to the same payee in the last 1 hour
    d_R   min hops (<=2) to a reported account, time-sliced graph
    v_t   pair velocity = past payments sender -> receiver in the last 24 h
    zc_t  cumulative unusual amount = log(amount + past 24 h sender -> receiver
          total) - median of past log amounts (defeats amount splitting)
    ptr_t pass-through = outflow / inflow of the account in the last 24 h
          (mule accounts forward what they receive; NaN when no inflow)

Each feature is computed for both endpoints (sender and receiver) and then
combined as the more anomalous value. Output: data/features.csv with is_scam,
plus non-feature helper columns (`pair24`, `in24_*`, `out24_*`: the raw 24 h
sums behind v_t/zc_t/ptr_t) that attack_sim uses to recompute them exactly.
24 h window sums use math.fsum so the brute-force check matches bit-for-bit.

Run:  python3 src/features.py --smoke   (reads/writes data/smoke/)
      python3 src/features.py
"""

from __future__ import annotations

import argparse
import math
import time
from bisect import bisect_left, bisect_right, insort
from pathlib import Path

import numpy as np
import pandas as pd

SMOKE_DIR = "data/smoke"  # written by journey_generator.py --smoke

DAY = 86400
W7 = 7 * DAY
W14 = 14 * DAY
H1 = 3600
CAP = 3

TX_COLS = ["txn_id", "ts", "day", "split", "sender_id", "receiver_id", "amount", "is_scam"]


def load_inputs(txn_path: str, edge_path: str, label_path: str):
    tx = pd.read_csv(txn_path, usecols=TX_COLS)
    tx["ts"] = pd.to_datetime(tx["ts"])
    tx["ts_int"] = (tx["ts"].astype("int64") // 1_000_000_000).astype(np.int64)
    tx = tx.sort_values("ts_int", kind="stable").reset_index(drop=True)

    labels = pd.read_csv(label_path)
    edges = pd.read_csv(edge_path, usecols=["src_vpa", "dst_vpa", "first_ts"])
    return tx, edges, labels


def endpoint_features(tx: pd.DataFrame) -> pd.DataFrame:
    n = len(tx)
    snd = tx["sender_id"].to_numpy(np.int64)
    rcv = tx["receiver_id"].to_numpy(np.int64)
    ts = tx["ts_int"].to_numpy(np.int64)
    amt = tx["amount"].to_numpy(np.float64)
    tid = tx["txn_id"].to_numpy(np.int64)

    acct = np.concatenate([snd, rcv])
    cp = np.concatenate([rcv, snd])
    tsv = np.concatenate([ts, ts])
    amt = np.concatenate([amt, amt])
    txid = np.concatenate([tid, tid])
    dir_in = np.concatenate([np.zeros(n, np.int8), np.ones(n, np.int8)])
    side = np.concatenate([np.zeros(n, np.int8), np.ones(n, np.int8)])
    lx = np.log(amt)

    order = np.lexsort((tsv, acct))
    acct = acct[order]
    cp = cp[order]
    tsv = tsv[order]
    amt = amt[order]
    txid = txid[order]
    dir_in = dir_in[order]
    side = side[order]
    lx = lx[order]

    L = acct.size
    f_n = np.zeros(L, np.float64)
    f_e = np.full(L, np.nan, np.float64)
    f_g = np.zeros(L, np.float64)
    f_z = np.full(L, np.nan, np.float64)
    f_p = np.zeros(L, np.float64)
    f_v = np.zeros(L, np.float64)
    f_pair = np.zeros(L, np.float64)
    f_zc = np.full(L, np.nan, np.float64)
    f_in = np.zeros(L, np.float64)
    f_out = np.zeros(L, np.float64)

    def window_sum(ts_list, amt_list, lo, hi):
        """(count, fsum) of events with lo < ts < hi."""
        a, b = bisect_right(ts_list, lo), bisect_left(ts_list, hi)
        return b - a, math.fsum(amt_list[a:b])

    bounds = np.concatenate([[0], np.flatnonzero(np.diff(acct)) + 1, [L]])
    for gi in range(len(bounds) - 1):
        st, en = int(bounds[gi]), int(bounds[gi + 1])
        logs: list = []
        ft: list = []
        fa: list = []
        credits: list = []
        small: dict = {}
        seen: set = set()
        # (counterparty, direction) -> ts / amounts, for the same-direction pair
        # flow; plus the account's own inflow / outflow history.
        pair_ts: dict = {}
        pair_amt: dict = {}
        flow_ts = ([], [])   # [0] = outgoing, [1] = incoming
        flow_amt = ([], [])
        j = st
        while j < en:
            T = int(tsv[j])
            k = j
            while k < en and tsv[k] == T:
                k += 1
            b = bisect_left(ft, T)
            nwin = b - bisect_right(ft, T - W7)
            if b:
                a = bisect_right(ft, T - W14)
                denom = max(fa[a:b]) if b > a else 0.0
                has_denom = b > a and denom > 0
            else:
                denom = 0.0
                has_denom = False
            gwin = credits and (bisect_left(credits, T) - bisect_right(credits, T - W7) > 0)
            m = len(logs)
            med = (logs[m >> 1] if (m & 1) else 0.5 * (logs[m // 2 - 1] + logs[m // 2])) if m else 0.0
            for idx in range(j, k):
                c = cp[idx]
                f_n[idx] = nwin
                if has_denom:
                    f_e[idx] = amt[idx] / denom
                if m:
                    f_z[idx] = lx[idx] - med
                lst = small.get(c)
                if lst is not None:
                    f_p[idx] = bisect_left(lst, T) - bisect_right(lst, T - H1)
                if gwin:
                    f_g[idx] = 1.0
                key = (c, dir_in[idx])
                plst = pair_ts.get(key)
                if plst is not None:
                    f_v[idx], f_pair[idx] = window_sum(plst, pair_amt[key], T - DAY, T)
                if m:
                    f_zc[idx] = math.log(amt[idx] + f_pair[idx]) - med
                f_out[idx] = window_sum(flow_ts[0], flow_amt[0], T - DAY, T)[1]
                f_in[idx] = window_sum(flow_ts[1], flow_amt[1], T - DAY, T)[1]
            for idx in range(j, k):
                c = cp[idx]
                key = (c, dir_in[idx])
                if key in pair_ts:
                    pair_ts[key].append(T)
                    pair_amt[key].append(amt[idx])
                else:
                    pair_ts[key] = [T]
                    pair_amt[key] = [amt[idx]]
                flow_ts[dir_in[idx]].append(T)
                flow_amt[dir_in[idx]].append(amt[idx])
                insort(logs, lx[idx])
                if c not in seen:
                    seen.add(c)
                    ft.append(T)
                    fa.append(amt[idx])
                    if dir_in[idx]:
                        credits.append(T)
                if amt[idx] <= 10:
                    lst = small.get(c)
                    if lst is None:
                        small[c] = [T]
                    else:
                        lst.append(T)
            j = k

    with np.errstate(divide="ignore", invalid="ignore"):
        f_ptr = np.where(f_in > 0, f_out / f_in, np.nan)
    ev = pd.DataFrame({
        "txn_id": txid, "side": side, "n_t": f_n, "e_t": f_e,
        "g_t": f_g, "z_t": f_z, "pi_t": f_p,
        "v_t": f_v, "zc_t": f_zc, "ptr_t": f_ptr,
        "pair24": f_pair, "in24": f_in, "out24": f_out,
    })
    return ev


def reported_proximity(tx: pd.DataFrame, edges: pd.DataFrame, labels: pd.DataFrame,
                       acc_count: int, n_days: int, start_ts: int) -> tuple:
    vmap = {v: i for i, v in enumerate(labels["vpa"].to_numpy())}
    src = edges["src_vpa"].map(vmap).to_numpy(np.int64)
    dst = edges["dst_vpa"].map(vmap).to_numpy(np.int64)
    epoch = (start_ts // DAY) * DAY
    fday = ((pd.to_datetime(edges["first_ts"]).astype("int64") // 1_000_000_000) - epoch).to_numpy(np.int64) // DAY
    eorder = np.argsort(fday, kind="stable")
    src, dst, fday = src[eorder], dst[eorder], fday[eorder]

    rep = labels[labels["is_reported"] == 1]
    rep_id = rep.index.to_numpy(np.int64)
    rep_day = rep["report_day"].to_numpy(np.float64)
    rorder = np.argsort(rep_day, kind="stable")
    rep_id, rep_day = rep_id[rorder], rep_day[rorder]

    adj: dict = {}
    dR_snd = np.full(len(tx), CAP, np.int8)
    dR_rcv = np.full(len(tx), CAP, np.int8)
    snd = tx["sender_id"].to_numpy(np.int64)
    rcv = tx["receiver_id"].to_numpy(np.int64)
    day = tx["day"].to_numpy(np.int64)

    edge_ptr = 0
    rep_ptr = 0
    active: set = set()
    dist = np.full(acc_count, CAP, np.int8)
    n_edges = len(src)
    for d in range(int(day.max()) + 1):
        while edge_ptr < n_edges and fday[edge_ptr] < d:
            u, v = int(src[edge_ptr]), int(dst[edge_ptr])
            adj.setdefault(u, set()).add(v)
            edge_ptr += 1
        while rep_ptr < len(rep_id) and rep_day[rep_ptr] < d:
            active.add(int(rep_id[rep_ptr]))
            rep_ptr += 1

        dist.fill(CAP)
        for r in active:
            dist[r] = 0
        n1: set = set()
        for r in active:
            n1 |= adj.get(r, set())
        n2: set = set()
        for u in n1:
            if dist[u] > 1:
                dist[u] = 1
            n2 |= adj.get(u, set())
        for w in n2:
            if dist[w] > 2:
                dist[w] = 2

        m = day == d
        if m.any():
            dR_snd[m] = dist[snd[m]]
            dR_rcv[m] = dist[rcv[m]]
    return dR_snd, dR_rcv


def combine(ev: pd.DataFrame, tx: pd.DataFrame, dR_snd: np.ndarray, dR_rcv: np.ndarray,
            is_business: np.ndarray) -> pd.DataFrame:
    idx = tx["txn_id"].to_numpy(np.int64)
    snd = ev[ev["side"] == 0].set_index("txn_id")
    rcv = ev[ev["side"] == 1].set_index("txn_id")

    def col(df, name):
        return df[name].reindex(idx).to_numpy(np.float64)

    ns, nr = col(snd, "n_t"), col(rcv, "n_t")
    es, er = col(snd, "e_t"), col(rcv, "e_t")
    gs, gr = col(snd, "g_t"), col(rcv, "g_t")
    zs, zr = col(snd, "z_t"), col(rcv, "z_t")
    ps, pr = col(snd, "pi_t"), col(rcv, "pi_t")
    vs, vr = col(snd, "v_t"), col(rcv, "v_t")
    zcs, zcr = col(snd, "zc_t"), col(rcv, "zc_t")
    pts, ptr_ = col(snd, "ptr_t"), col(rcv, "ptr_t")

    def larger_abs(a, b):
        return np.where(np.isnan(a), b,
                        np.where(np.isnan(b), a,
                                 np.where(np.abs(a) >= np.abs(b), a, b)))

    z = larger_abs(zs, zr)

    sb = is_business[tx["sender_id"].to_numpy(np.int64)]
    rb = is_business[tx["receiver_id"].to_numpy(np.int64)]
    only_s = ~sb & rb
    only_r = sb & ~rb

    def user_side(a, b):
        return np.where(only_s, a, np.where(only_r, b, np.fmax(a, b)))

    out = pd.DataFrame({
        "txn_id": idx,
        "ts": tx["ts"].to_numpy(),
        "day": tx["day"].to_numpy(),
        "split": tx["split"].to_numpy(),
        "sender_id": tx["sender_id"].to_numpy(),
        "receiver_id": tx["receiver_id"].to_numpy(),
        "amount": tx["amount"].to_numpy(),
        "n_t": user_side(ns, nr),
        "e_t": user_side(es, er),
        "g_t": np.maximum(gs, gr).astype(np.int8),
        "z_t": z,
        "pi_t": np.maximum(ps, pr),
        "d_R": np.minimum(dR_snd, dR_rcv).astype(np.int8),
        "n_t_snd": ns, "n_t_rcv": nr,
        "e_t_snd": es, "e_t_rcv": er,
        "g_t_snd": gs, "g_t_rcv": gr,
        "z_t_snd": zs, "z_t_rcv": zr,
        "pi_t_snd": ps, "pi_t_rcv": pr,
        "d_R_snd": dR_snd, "d_R_rcv": dR_rcv,
        "v_t": np.maximum(vs, vr),
        "zc_t": larger_abs(zcs, zcr),
        "ptr_t": np.fmax(pts, ptr_),
        "v_t_snd": vs, "v_t_rcv": vr,
        "zc_t_snd": zcs, "zc_t_rcv": zcr,
        "ptr_t_snd": pts, "ptr_t_rcv": ptr_,
        # helper columns (not model features): raw 24 h sums for attack_sim
        "pair24": col(snd, "pair24"),
        "in24_snd": col(snd, "in24"), "out24_snd": col(snd, "out24"),
        "in24_rcv": col(rcv, "in24"), "out24_rcv": col(rcv, "out24"),
        "is_scam": tx["is_scam"].to_numpy(),
    })
    return out


def brute_force(tx: pd.DataFrame, sample: pd.DataFrame, is_business: np.ndarray) -> pd.DataFrame:
    endpoints = set(sample["sender_id"]) | set(sample["receiver_id"])
    sub = tx[tx["sender_id"].isin(endpoints) | tx["receiver_id"].isin(endpoints)].copy()
    rows = []
    for row in sample.itertuples(index=False):
        s, r, x, T = int(row.sender_id), int(row.receiver_id), float(row.amount), int(row.ts_int)
        vals = {}
        for side, e, cp in (("snd", s, r), ("rcv", r, s)):
            inv = sub[(sub["sender_id"] == e) | (sub["receiver_id"] == e)]
            other = np.where(inv["sender_id"].to_numpy() == e, inv["receiver_id"].to_numpy(),
                             inv["sender_id"].to_numpy())
            inv = inv.assign(cp=other)
            first = inv.groupby("cp")["ts_int"].min()
            first_evt = inv.sort_values("ts_int", kind="stable").groupby("cp").first()
            first_amt = first_evt["amount"]
            it = first.to_numpy()
            n_t = int(((it > T - W7) & (it < T)).sum())
            sel = (it > T - W14) & (it < T)
            if sel.any():
                e_t = x / float(first_amt.to_numpy()[sel].max())
            else:
                e_t = np.nan
            past = inv[inv["ts_int"] < T]
            z_t = float(np.log(x) - np.median(np.log(past["amount"].to_numpy()))) if len(past) else np.nan
            pair_mask = (((past["sender_id"] == e) & (past["receiver_id"] == cp)) |
                         ((past["sender_id"] == cp) & (past["receiver_id"] == e)))
            pi = int((pair_mask.to_numpy() & (past["amount"] <= 10).to_numpy()
                      & (past["ts_int"] > T - H1).to_numpy()).sum())
            win = past[past["ts_int"] > T - DAY]
            pair = win[(win["sender_id"] == s) & (win["receiver_id"] == r)]["amount"]
            v_t, pair_sum = len(pair), math.fsum(pair)
            zc_t = (math.log(x + pair_sum) - float(np.median(np.log(past["amount"].to_numpy())))
                    if len(past) else np.nan)
            in24 = math.fsum(win[win["receiver_id"] == e]["amount"])
            out24 = math.fsum(win[win["sender_id"] == e]["amount"])
            ptr_t = out24 / in24 if in24 > 0 else np.nan
            credits = past[past["receiver_id"] == e]
            g = 0
            for cr in credits.itertuples(index=False):
                if cr.ts_int > T - W7 and int(first.get(cr.sender_id, -1)) == int(cr.ts_int):
                    g = 1
                    break
            vals[side] = dict(n_t=n_t, e_t=e_t, z_t=z_t, pi_t=pi, g_t=g,
                              v_t=v_t, zc_t=zc_t, ptr_t=ptr_t)
        s_v, r_v = vals["snd"], vals["rcv"]
        sb, rb = bool(is_business[s]), bool(is_business[r])
        if (not sb) and rb:
            n_t, e_t = s_v["n_t"], s_v["e_t"]
        elif sb and (not rb):
            n_t, e_t = r_v["n_t"], r_v["e_t"]
        else:
            n_t, e_t = max(s_v["n_t"], r_v["n_t"]), np.fmax(s_v["e_t"], r_v["e_t"])
        def larger_abs(a, b):
            return a if (np.isnan(b) or (not np.isnan(a) and abs(a) >= abs(b))) else b

        z = larger_abs(s_v["z_t"], r_v["z_t"])
        rows.append(dict(
            v_t=max(s_v["v_t"], r_v["v_t"]),
            zc_t=larger_abs(s_v["zc_t"], r_v["zc_t"]),
            ptr_t=np.fmax(s_v["ptr_t"], r_v["ptr_t"]),
            txn_id=row.txn_id,
            n_t=n_t,
            e_t=e_t,
            g_t=max(s_v["g_t"], r_v["g_t"]),
            z_t=z,
            pi_t=max(s_v["pi_t"], r_v["pi_t"]),
        ))
    return pd.DataFrame(rows)


def validate(tx: pd.DataFrame, feats: pd.DataFrame, n_days: int, split_fracs: tuple,
             is_business: np.ndarray) -> dict:
    report = {}
    f = feats.set_index("txn_id")
    samp = tx.sample(n=min(300, len(tx)), random_state=1)
    bf = brute_force(tx, samp, is_business)
    bf = bf.set_index("txn_id")
    common = bf.index
    checks = {}
    for c in ["n_t", "pi_t", "g_t", "v_t"]:
        checks[c] = float(np.abs(f.loc[common, c].to_numpy() - bf[c].to_numpy()).max())
    for c in ["e_t", "z_t", "zc_t", "ptr_t"]:
        a = f.loc[common, c].to_numpy()
        b = bf[c].to_numpy()
        both_nan = np.isnan(a) & np.isnan(b)
        diff = np.where(both_nan, 0.0, np.abs(np.nan_to_num(a) - np.nan_to_num(b)))
        diff = np.where(np.isnan(a) ^ np.isnan(b), np.inf, diff)
        checks[c] = float(diff.max())
    report["max_abs_diff_sample"] = checks
    report["is_scam_rate"] = float(feats["is_scam"].mean())
    report["scam_stats"] = feats.groupby("split")["is_scam"].agg(["size", "sum", "mean"]).to_dict()
    report["feature_nan_frac"] = {c: float(feats[c].isna().mean())
                                  for c in ["n_t", "e_t", "g_t", "z_t", "pi_t", "d_R", "v_t", "zc_t", "ptr_t"]}
    report["d_R_dist"] = feats["d_R"].value_counts().sort_index().to_dict()
    report["rows"] = int(len(feats))
    return report


def run(args) -> None:
    t0 = time.time()
    tx, edges, labels = load_inputs(args.txns, args.edges, args.labels)
    start_ts = int(tx["ts_int"].min())
    acc_count = int(max(tx["sender_id"].max(), tx["receiver_id"].max())) + 1

    role = labels.set_index("account_id")["role"]
    is_business = role.reindex(np.arange(acc_count)).eq("business").to_numpy()

    ev = endpoint_features(tx)
    dR_snd, dR_rcv = reported_proximity(tx, edges, labels, acc_count, args.days, start_ts)
    feats = combine(ev, tx, dR_snd, dR_rcv, is_business)

    report = validate(tx, feats, args.days, (0.6, 0.2, 0.2), is_business)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out, index=False)
    # Fast-load copy; risk_common.load_features() prefers it while it is fresh.
    feats.to_parquet(out.with_suffix(".parquet"), index=False)

    print(f"[features] rows={len(feats):,} accounts={acc_count} scam_rate={report['is_scam_rate']:.5f}")
    print(f"[features] sample max|diff|: {report['max_abs_diff_sample']}")
    print(f"[features] nan_frac: {report['feature_nan_frac']}")
    print(f"[features] d_R dist: {report['d_R_dist']}")
    print(f"[features] wrote {out.resolve()} (+ .parquet) ({time.time() - t0:.1f}s)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Build past-only fraud features.")
    ap.add_argument("--txns", default=None, help="default: <data dir>/transactions.csv")
    ap.add_argument("--edges", default=None, help="default: <data dir>/graph_edges.csv")
    ap.add_argument("--labels", default=None, help="default: <data dir>/labels.csv")
    ap.add_argument("--out", default=None, help="default: <data dir>/features.csv")
    ap.add_argument("--days", type=int, default=None, help="default: 120 (30 with --smoke)")
    ap.add_argument("--smoke", action="store_true",
                    help="use data/smoke/ (from journey_generator.py --smoke) instead of data/")
    args = ap.parse_args(argv)
    data_dir = Path(SMOKE_DIR if args.smoke else "data")
    args.txns = args.txns or str(data_dir / "transactions.csv")
    args.edges = args.edges or str(data_dir / "graph_edges.csv")
    args.labels = args.labels or str(data_dir / "labels.csv")
    args.out = args.out or str(data_dir / "features.csv")
    if args.days is None:
        args.days = 30 if args.smoke else 120
    run(args)


if __name__ == "__main__":
    main()
