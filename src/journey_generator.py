"""Synthetic UPI transaction / scam-journey generator for the maya project.

Generates normal UPI activity (salary, rent, weddings, P2P, merchant) plus three
scam types (investment, digital arrest, sextortion), each as a five-stage journey,
routed through layered mule networks. Emits three datasets to data/:
    transactions.csv, graph_edges.csv, labels.csv

Split by contiguous day blocks (train/val/test) to prevent temporal leakage.
Run:  python3 src/journey_generator.py --smoke   (writes to data/smoke/)
      python3 src/journey_generator.py
"""

from __future__ import annotations

import argparse
import os
import time
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

SMOKE_DIR = "data/smoke"  # --smoke output; features.py --smoke reads it

CATEGORIES = [
    "salary", "rent", "wedding", "p2p", "merchant",
    "investment", "digital_arrest", "sextortion", "layering",
]
CAT_SALARY, CAT_RENT, CAT_WEDDING, CAT_P2P, CAT_MERCHANT = 0, 1, 2, 3, 4
CAT_INVEST, CAT_ARREST, CAT_SEXTO, CAT_LAYER = 5, 6, 7, 8

SCAM_TYPES = ["investment", "digital_arrest", "sextortion"]
SCAM_CAT = {"investment": CAT_INVEST, "digital_arrest": CAT_ARREST, "sextortion": CAT_SEXTO}

TXN_TYPES = ["P2P", "P2M"]
STATUSES = ["SUCCESS", "FAILED", "PENDING"]
APPS = ["phonepe", "gpay", "paytm", "bhim"]
BANKS = ["okaxis", "oksbi", "okhdfcbank", "ybl", "paytm", "ibl", "axl"]

SPLITS = ["train", "val", "test"]

SCAM_PLANS = {
    "investment": [
        (1, "receive", 500, 5000, 1, 0, 0),
        (2, "payout", 200, 2000, 2, 2 * 86400, 4 * 86400),
        (3, "receive", 5000, 100000, 2, 1 * 86400, 3 * 86400),
        (4, "receive", 20000, 400000, 2, 3 * 86400, 5 * 86400),
    ],
    "digital_arrest": [
        (1, "receive", 10000, 50000, 1, 0, 0),
        (2, "receive", 50000, 200000, 1, 1200, 5400),
        (3, "receive", 200000, 800000, 1, 1200, 5400),
        (4, "receive", 100000, 600000, 1, 1200, 5400),
    ],
    "sextortion": [
        (1, "receive", 1000, 10000, 1, 0, 0),
        (2, "receive", 2000, 30000, 1, 6 * 3600, 30 * 3600),
        (3, "receive", 5000, 60000, 2, 6 * 3600, 24 * 3600),
        (4, "receive", 10000, 80000, 3, 6 * 3600, 24 * 3600),
    ],
}
SCAM_DURATION_DAYS = {"investment": 16, "digital_arrest": 2, "sextortion": 12}
STAGE_REPORT_WEIGHT = {1: 0.2, 2: 0.35, 3: 0.7, 4: 0.85, 5: 0.9}


@dataclass
class Config:
    seed: int = 42
    n_users: int = 20_000
    n_days: int = 120
    split_fracs: tuple = (0.6, 0.2, 0.2)
    out_dir: str = "data"
    start_date: str = "2024-01-01"

    victim_rate: float = 0.015
    mule_rate: float = 0.003

    p2p_lambda: float = 1.0
    merchant_lambda: float = 0.45
    friend_k: int = 6

    n_employers: int = 150
    n_landlords: int = 400
    n_merchants: int = 800
    n_aggregators: int = 8
    n_exits: int = 4

    salary_day_max: int = 5
    rent_day_max: int = 10

    wedding_event_rate: float = 0.04
    wedding_gifts: int = 18

    report_prob: float = 0.75
    report_delay_min: float = 1.0
    report_delay_max: float = 60.0
    report_coverage_a: float = 2.0
    report_coverage_b: float = 2.5

    scam_mix: tuple = (0.45, 0.25, 0.30)

    smoke: bool = False
    write_csv: bool = True

    def resolved(self) -> "Config":
        if not self.smoke:
            return self
        return replace(
            self,
            n_users=600,
            n_days=30,
            n_employers=20,
            n_landlords=40,
            n_merchants=60,
            n_aggregators=4,
            n_exits=2,
            mule_rate=0.01,
            victim_rate=0.03,
        )


def make_rng(seed: int, name: str) -> np.random.Generator:
    key = zlib.crc32(name.encode("utf-8"))
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(key,)))


def split_codes(n_days: int, fracs: tuple) -> np.ndarray:
    f = np.asarray(fracs, dtype=float)
    f = f / f.sum()
    n_train = int(n_days * f[0])
    n_val = int(n_days * f[1])
    n_train = max(1, min(n_train, n_days - 2))
    n_val = max(1, min(n_val, n_days - n_train - 1))
    codes = np.zeros(n_days, dtype=np.int8)
    codes[n_train:n_train + n_val] = 1
    codes[n_train + n_val:] = 2
    return codes


def block_bounds(n_days: int, fracs: tuple) -> dict:
    f = np.asarray(fracs, dtype=float)
    f = f / f.sum()
    n_train = int(n_days * f[0])
    n_val = int(n_days * f[1])
    n_train = max(1, min(n_train, n_days - 2))
    n_val = max(1, min(n_val, n_days - n_train - 1))
    return {"train": (0, n_train), "val": (n_train, n_train + n_val), "test": (n_train + n_val, n_days)}


@dataclass
class Accounts:
    vpa: list
    role: np.ndarray
    join_day: np.ndarray
    user_id: np.ndarray
    n_users: int
    n_accounts: int
    blocks: np.ndarray = field(default=None)
    emp: tuple = (0, 0)
    landlord: tuple = (0, 0)
    merchant: tuple = (0, 0)
    agg: tuple = (0, 0)
    exit_: tuple = (0, 0)


def build_accounts(cfg: Config, rng: np.random.Generator) -> Accounts:
    n = cfg.n_users
    vpas = [f"u{i:06d}@{BANKS[i % len(BANKS)]}" for i in range(n)]
    roles = np.empty(n, dtype=object)
    roles[:] = "user"
    user_id = np.arange(n)
    join_day = rng.integers(0, max(1, cfg.n_days // 3), size=n).astype(np.int32)
    join_day = np.clip(join_day, 0, cfg.n_days - 1)

    cur = n
    emp0 = cur
    vpas += [f"emp{i}@biz" for i in range(cfg.n_employers)]
    roles = np.concatenate([roles, np.full(cfg.n_employers, "business", dtype=object)])
    user_id = np.concatenate([user_id, np.full(cfg.n_employers, -1, dtype=np.int64)])
    join_day = np.concatenate([join_day, np.zeros(cfg.n_employers, dtype=np.int32)])
    cur += cfg.n_employers
    emp1 = cur

    ll0 = cur
    vpas += [f"ll{i}@biz" for i in range(cfg.n_landlords)]
    roles = np.concatenate([roles, np.full(cfg.n_landlords, "business", dtype=object)])
    user_id = np.concatenate([user_id, np.full(cfg.n_landlords, -1, dtype=np.int64)])
    join_day = np.concatenate([join_day, np.zeros(cfg.n_landlords, dtype=np.int32)])
    cur += cfg.n_landlords
    ll1 = cur

    mc0 = cur
    vpas += [f"mch{i}@merchant" for i in range(cfg.n_merchants)]
    roles = np.concatenate([roles, np.full(cfg.n_merchants, "business", dtype=object)])
    user_id = np.concatenate([user_id, np.full(cfg.n_merchants, -1, dtype=np.int64)])
    join_day = np.concatenate([join_day, np.zeros(cfg.n_merchants, dtype=np.int32)])
    cur += cfg.n_merchants
    mc1 = cur

    ag0 = cur
    vpas += [f"agg{i}@acct" for i in range(cfg.n_aggregators)]
    roles = np.concatenate([roles, np.full(cfg.n_aggregators, "aggregator", dtype=object)])
    user_id = np.concatenate([user_id, np.full(cfg.n_aggregators, -1, dtype=np.int64)])
    join_day = np.concatenate([join_day, np.zeros(cfg.n_aggregators, dtype=np.int32)])
    cur += cfg.n_aggregators
    ag1 = cur

    ex0 = cur
    vpas += [f"out{i}@acct" for i in range(cfg.n_exits)]
    roles = np.concatenate([roles, np.full(cfg.n_exits, "exit", dtype=object)])
    user_id = np.concatenate([user_id, np.full(cfg.n_exits, -1, dtype=np.int64)])
    join_day = np.concatenate([join_day, np.zeros(cfg.n_exits, dtype=np.int32)])
    cur += cfg.n_exits
    ex1 = cur

    return Accounts(
        vpa=vpas, role=roles, join_day=join_day, user_id=user_id, n_users=n, n_accounts=cur,
        emp=(emp0, emp1), landlord=(ll0, ll1), merchant=(mc0, mc1), agg=(ag0, ag1), exit_=(ex0, ex1),
    )


@dataclass
class Users:
    income: np.ndarray
    rent: np.ndarray
    salary_day: np.ndarray
    rent_day: np.ndarray
    friends: np.ndarray
    employer: np.ndarray
    landlord: np.ndarray


def generate_users(cfg: Config, acc: Accounts, rng: np.random.Generator) -> Users:
    n = cfg.n_users
    bracket = rng.choice([0, 1, 2], size=n, p=[0.55, 0.30, 0.15])
    base = np.array([25000.0, 70000.0, 200000.0])[bracket]
    income = base * rng.lognormal(0.0, 0.35, size=n)
    salary_day = rng.integers(1, cfg.salary_day_max + 1, size=n)
    rent_day = rng.integers(1, cfg.rent_day_max + 1, size=n)
    rent = income * rng.uniform(0.15, 0.35, size=n)
    employer = rng.integers(acc.emp[0], acc.emp[1], size=n)
    landlord = rng.integers(acc.landlord[0], acc.landlord[1], size=n)

    friends = rng.integers(0, n, size=(n, cfg.friend_k)).astype(np.int64)
    self_mask = friends == np.arange(n)[:, None]
    friends[self_mask] = (friends[self_mask] + 1) % n
    return Users(income=income, rent=rent, salary_day=salary_day, rent_day=rent_day,
                 friends=friends, employer=employer, landlord=landlord)


class Ledger:
    def __init__(self) -> None:
        self.fields: dict[str, list] = {k: [] for k in (
            "sender", "receiver", "day", "tsec", "amount", "cat", "ttype",
            "app", "status", "campaign", "scam_type", "stage", "is_scam", "is_in_report",
        )}

    def add(self, **kw) -> None:
        n = len(kw["sender"])
        for k in self.fields:
            self.fields[k].append(kw[k] if k in kw else
                                  (-np.ones(n, dtype=np.int64) if k in ("campaign", "scam_type") else
                                   np.zeros(n, dtype=np.int64)))

    def add_row(self, sender, receiver, day, tsec, amount, cat, ttype, app, status,
                campaign=-1, scam_type=-1, stage=0, is_scam=0, is_in_report=0) -> None:
        self.add(
            sender=np.array([sender], dtype=np.int64), receiver=np.array([receiver], dtype=np.int64),
            day=np.array([day], dtype=np.int64), tsec=np.array([tsec], dtype=np.int64),
            amount=np.array([amount], dtype=np.float64), cat=np.array([cat], dtype=np.int64),
            ttype=np.array([ttype], dtype=np.int64), app=np.array([app], dtype=np.int64),
            status=np.array([status], dtype=np.int64), campaign=np.array([campaign], dtype=np.int64),
            scam_type=np.array([scam_type], dtype=np.int64), stage=np.array([stage], dtype=np.int64),
            is_scam=np.array([is_scam], dtype=np.int64), is_in_report=np.array([is_in_report], dtype=np.int64),
        )


def add_normal(cfg: Config, acc: Accounts, us: Users, rng: np.random.Generator, led: Ledger) -> None:
    n, D = cfg.n_users, cfg.n_days
    days = np.arange(D)
    active = days[None, :] >= acc.join_day[:n, None]

    counts = rng.poisson(cfg.p2p_lambda, size=(n, D)).astype(np.int64)
    counts[~active] = 0
    flat = counts.reshape(-1)
    tot = int(flat.sum())
    if tot:
        pair_user = np.repeat(np.arange(n), D)
        pair_day = np.tile(days, n)
        send = np.repeat(pair_user, flat)
        day = np.repeat(pair_day, flat)
        recv = us.friends[send, rng.integers(0, cfg.friend_k, size=tot)]
        amount = np.clip(np.exp(rng.normal(6.0, 1.1, size=tot)), 10, 200000)
        probe = rng.random(tot) < 0.02
        if probe.any():
            amount[probe] = rng.choice([1.0, 5.0, 10.0], size=int(probe.sum()))
        sec = rng.integers(0, 86400, size=tot)
        led.add(sender=send, receiver=recv, day=day, tsec=day * 86400 + sec, amount=amount,
                cat=np.full(tot, CAT_P2P), ttype=np.zeros(tot, np.int64),
                app=rng.integers(0, len(APPS), tot), status=rng.choice([0, 1, 2], size=tot, p=[0.95, 0.04, 0.01]))

    mcounts = rng.poisson(cfg.merchant_lambda, size=(n, D)).astype(np.int64)
    mcounts[~active] = 0
    mflat = mcounts.reshape(-1)
    mtot = int(mflat.sum())
    if mtot:
        pair_user = np.repeat(np.arange(n), D)
        pair_day = np.tile(days, n)
        send = np.repeat(pair_user, mflat)
        day = np.repeat(pair_day, mflat)
        recv = rng.integers(acc.merchant[0], acc.merchant[1], size=mtot)
        amount = np.clip(np.exp(rng.normal(5.2, 1.0, size=mtot)), 10, 100000)
        sec = rng.integers(0, 86400, size=mtot)
        led.add(sender=send, receiver=recv, day=day, tsec=day * 86400 + sec, amount=amount,
                cat=np.full(mtot, CAT_MERCHANT), ttype=np.ones(mtot, np.int64),
                app=rng.integers(0, len(APPS), mtot), status=rng.choice([0, 1, 2], size=mtot, p=[0.96, 0.03, 0.01]))

    sal_mask = ((days[None, :] % 30 == us.salary_day[:, None]) & (days[None, :] >= 1) & active)
    ui, di = np.where(sal_mask)
    if ui.size:
        amount = us.income[ui] * rng.lognormal(0.0, 0.05, size=ui.size)
        sec = rng.integers(0, 6 * 3600, size=ui.size)
        led.add(sender=us.employer[ui], receiver=ui, day=di, tsec=di * 86400 + sec, amount=amount,
                cat=np.full(ui.size, CAT_SALARY), ttype=np.ones(ui.size, np.int64),
                app=rng.integers(0, len(APPS), ui.size), status=np.zeros(ui.size, np.int64))

    rent_mask = ((days[None, :] % 30 == us.rent_day[:, None]) & (days[None, :] >= 1) & active)
    ui, di = np.where(rent_mask)
    if ui.size:
        amount = us.rent[ui] * rng.lognormal(0.0, 0.03, size=ui.size)
        sec = rng.integers(6 * 3600, 12 * 3600, size=ui.size)
        led.add(sender=ui, receiver=us.landlord[ui], day=di, tsec=di * 86400 + sec, amount=amount,
                cat=np.full(ui.size, CAT_RENT), ttype=np.zeros(ui.size, np.int64),
                app=rng.integers(0, len(APPS), ui.size), status=np.zeros(ui.size, np.int64))

    wrng = make_rng(cfg.seed, "weddings")
    hosts = np.where((wrng.random(n) < cfg.wedding_event_rate) & (acc.join_day[:n] < D - 2))[0]
    for h in hosts:
        d0 = int(acc.join_day[h]) + 1
        d1 = min(D - 1, d0 + 2)
        if d1 <= d0:
            continue
        eday = int(wrng.integers(d0, d1 + 1))
        g = cfg.wedding_gifts
        for _ in range(g):
            friend = int(us.friends[h, wrng.integers(0, cfg.friend_k)])
            amt = float(np.clip(np.exp(wrng.normal(8.0, 1.0)), 500, 50000))
            sec = int(wrng.integers(0, 86400))
            led.add_row(friend, h, eday, eday * 86400 + sec, amt, CAT_WEDDING, 0,
                        int(wrng.integers(0, len(APPS))), 0)
        for _ in range(int(wrng.integers(1, 4))):
            mch = int(wrng.integers(acc.merchant[0], acc.merchant[1]))
            amt = float(np.clip(np.exp(wrng.normal(9.5, 0.8)), 5000, 500000))
            sec = int(wrng.integers(0, 86400))
            led.add_row(h, mch, eday, eday * 86400 + sec, amt, CAT_WEDDING, 1,
                        int(wrng.integers(0, len(APPS))), 0)


@dataclass
class MuleNet:
    graph: nx.DiGraph
    entry: np.ndarray
    mid: np.ndarray
    agg: np.ndarray
    exits: np.ndarray


def build_mule_network(cfg: Config, acc: Accounts, users_idx: np.ndarray, rng: np.random.Generator) -> MuleNet:
    m = users_idx.size
    n_entry = max(1, int(round(0.65 * m)))
    entry = np.sort(users_idx[:n_entry])
    mid = np.sort(users_idx[n_entry:])
    if mid.size == 0:
        mid = entry.copy()
    agg = np.arange(acc.agg[0], acc.agg[1])
    exits = np.arange(acc.exit_[0], acc.exit_[1])
    G = nx.DiGraph()
    for e in entry:
        k = rng.integers(1, 3)
        for t in rng.choice(mid, size=min(int(k), mid.size), replace=False):
            G.add_edge(int(e), int(t))
    for node in mid:
        k = rng.integers(1, 3)
        for a in rng.choice(agg, size=min(int(k), agg.size), replace=False):
            G.add_edge(int(node), int(a))
    for a in agg:
        k = rng.integers(1, 3)
        for x in rng.choice(exits, size=min(int(k), exits.size), replace=False):
            G.add_edge(int(a), int(x))
    return MuleNet(graph=G, entry=entry, mid=mid, agg=agg, exits=exits)


@dataclass
class Campaign:
    idx: int
    scam_type: str
    split: int
    victim: int
    entry_mule: int
    mids_used: list
    agg_used: int
    exit_used: int
    last_ts: int
    n_scam_txns: int
    reported: bool
    report_day: float
    report_delay: float
    coverage: float


def generate_campaigns(cfg: Config, acc: Accounts, us: Users, net: MuleNet,
                       rng: np.random.Generator, led: Ledger) -> list:
    n, D = cfg.n_users, cfg.n_days
    blocks = block_bounds(D, cfg.split_fracs)
    n_victims = max(1, int(round(cfg.victim_rate * n)))
    pool = np.setdiff1d(np.arange(n), np.concatenate([net.entry, net.mid]))
    rng.shuffle(pool)
    victims = pool[:n_victims]

    mix = np.asarray(cfg.scam_mix, dtype=float)
    mix = mix / mix.sum()
    block_names = list(SPLITS)
    weights = np.array([blocks[b][1] - blocks[b][0] for b in block_names], dtype=float)
    weights = weights / weights.sum()

    campaigns = []
    for i, victim in enumerate(victims):
        stype = SCAM_TYPES[int(rng.choice(len(SCAM_TYPES), p=mix))]
        sidx = int(rng.choice(len(SPLITS), p=weights))
        d0, d1 = blocks[block_names[sidx]]
        dur = SCAM_DURATION_DAYS[stype]
        hi = max(d0 + 1, d1 - dur)
        start_day = int(rng.integers(d0, hi))

        entry = int(rng.choice(net.entry))
        succ_mid = list(net.graph.successors(entry))
        if not succ_mid:
            succ_mid = [int(rng.choice(net.mid))]
        mids_used = list(rng.choice(succ_mid, size=min(int(rng.integers(1, 3)), len(succ_mid)), replace=False))
        agg_succ = []
        for md in mids_used:
            agg_succ.extend(list(net.graph.successors(md)))
        if not agg_succ:
            agg_succ = [int(rng.choice(net.agg))]
        agg_used = int(rng.choice(agg_succ))
        exit_succ = list(net.graph.successors(agg_used)) or [int(rng.choice(net.exits))]
        exit_used = int(rng.choice(exit_succ))

        scale = float(np.clip(us.income[victim] / 60000.0, 0.6, 2.5))
        cur = start_day * 86400 + int(rng.integers(8 * 3600, 20 * 3600))
        gross_in = 0.0
        gross_out = 0.0
        plan = SCAM_PLANS[stype]
        rows = []
        for stage, direction, lo, hi_a, count, gap_lo, gap_hi in plan:
            for j in range(int(count)):
                if stage == 1 and j == 0:
                    pass
                else:
                    cur += int(rng.integers(int(gap_lo), int(gap_hi) + 1)) if gap_hi > 0 else 0
                amt = float(np.clip(rng.uniform(lo, hi_a) * scale, lo * 0.5, 1_500_000))
                rows.append((stage, direction, cur, amt))
                if direction == "receive":
                    gross_in += amt
                else:
                    gross_out += amt

        min_ts = min(r[2] for r in rows)
        max_ts = max(r[2] for r in rows)
        span = max_ts - min_ts
        block_len = d1 - d0
        max_span = max(0, (block_len - 1) * 86400)
        if span > max_span:
            new_start = d0 * 86400 + 8 * 3600
            factor = (max_span / span) if span > 0 else 0.0
            rows = [(s, d, new_start + int((t - min_ts) * factor), a) for (s, d, t, a) in rows]
            max_ts = max(r[2] for r in rows)
        else:
            overflow = (max_ts // 86400) - (d1 - 1)
            if overflow > 0:
                rows = [(s, d, t - overflow * 86400, a) for (s, d, t, a) in rows]
                max_ts -= overflow * 86400

        coverage = float(rng.beta(cfg.report_coverage_a, cfg.report_coverage_b))
        reported = bool(rng.random() < cfg.report_prob)
        report_delay = float(np.clip(rng.lognormal(np.log(6.0), 0.8), cfg.report_delay_min, cfg.report_delay_max))
        report_day = max_ts // 86400 + report_delay

        for stage, direction, t, amt in rows:
            day = int(t // 86400)
            sec = int(t - day * 86400)
            if direction == "receive":
                s, r = victim, entry
            else:
                s, r = entry, victim
            in_rep = int(reported and rng.random() < coverage * STAGE_REPORT_WEIGHT[stage])
            led.add_row(s, r, day, t, amt, SCAM_CAT[stype], 0, int(rng.integers(0, len(APPS))), 0,
                        campaign=i, scam_type=SCAM_TYPES.index(stype), stage=stage,
                        is_scam=1, is_in_report=in_rep)

        t = max_ts + int(rng.integers(1800, 86400))
        for md in mids_used:
            amt = max(1.0, gross_in * 0.90 / len(mids_used))
            day = int(t // 86400)
            if day >= d1:
                day = d1 - 1
                t = day * 86400 + 3600
            sec = int(t - day * 86400)
            led.add_row(entry, int(md), day, day * 86400 + sec, amt, CAT_LAYER, 0,
                        int(rng.integers(0, len(APPS))), 0, campaign=i,
                        scam_type=SCAM_TYPES.index(stype), stage=5, is_scam=1,
                        is_in_report=int(reported and rng.random() < coverage * STAGE_REPORT_WEIGHT[5]))
            t += int(rng.integers(600, 7200))
        lay1 = gross_in * 0.90
        for md in mids_used:
            amt = max(1.0, lay1 * 0.95 / len(mids_used))
            day = min(int(t // 86400), d1 - 1)
            sec = int(t - day * 86400) if int(t // 86400) < d1 else 3600
            led.add_row(int(md), agg_used, day, day * 86400 + sec, amt, CAT_LAYER, 0,
                        int(rng.integers(0, len(APPS))), 0, campaign=i,
                        scam_type=SCAM_TYPES.index(stype), stage=5, is_scam=1,
                        is_in_report=int(reported and rng.random() < coverage * STAGE_REPORT_WEIGHT[5]))
            t += int(rng.integers(600, 7200))
        amt = max(1.0, lay1 * 0.95 * 0.97)
        day = min(int(t // 86400), d1 - 1)
        sec = int(t - day * 86400) if int(t // 86400) < d1 else 3600
        led.add_row(agg_used, exit_used, day, day * 86400 + sec, amt, CAT_LAYER, 0,
                    int(rng.integers(0, len(APPS))), 0, campaign=i,
                    scam_type=SCAM_TYPES.index(stype), stage=5, is_scam=1,
                    is_in_report=int(reported and rng.random() < coverage * STAGE_REPORT_WEIGHT[5]))

        campaigns.append(Campaign(
            idx=i, scam_type=stype, split=sidx, victim=int(victim), entry_mule=entry,
            mids_used=[int(x) for x in mids_used], agg_used=agg_used, exit_used=exit_used,
            last_ts=int(max_ts), n_scam_txns=len(rows), reported=reported,
            report_day=report_day, report_delay=report_delay, coverage=coverage,
        ))
    return campaigns


def build_transactions(cfg: Config, led: Ledger, acc: Accounts, n_campaigns: int) -> pd.DataFrame:
    arrs = {k: np.concatenate(v) if v else np.array([], dtype=np.int64) for k, v in led.fields.items()}
    order = np.argsort(arrs["tsec"], kind="stable")
    for k in arrs:
        arrs[k] = arrs[k][order]
    n = arrs["sender"].size
    base = np.datetime64(cfg.start_date)
    ts = base + arrs["tsec"].astype("timedelta64[s]")
    camp_cats = [f"C{i:05d}" for i in range(n_campaigns)] if n_campaigns else []
    tx = pd.DataFrame({
        "txn_id": np.arange(n, dtype=np.int64),
        "ts": ts.astype("datetime64[ns]"),
        "day": arrs["day"].astype(np.int16),
        "sender_id": arrs["sender"].astype(np.int64),
        "sender_vpa": pd.Categorical.from_codes(arrs["sender"].astype(np.int64), categories=acc.vpa),
        "receiver_id": arrs["receiver"].astype(np.int64),
        "receiver_vpa": pd.Categorical.from_codes(arrs["receiver"].astype(np.int64), categories=acc.vpa),
        "amount": np.round(arrs["amount"], 2),
        "category": pd.Categorical.from_codes(arrs["cat"], categories=CATEGORIES),
        "txn_type": pd.Categorical.from_codes(arrs["ttype"], categories=TXN_TYPES),
        "app": pd.Categorical.from_codes(arrs["app"], categories=APPS),
        "status": pd.Categorical.from_codes(arrs["status"], categories=STATUSES),
        "campaign_id": pd.Categorical.from_codes(arrs["campaign"].astype(np.int64), categories=camp_cats),
        "scam_type": pd.Categorical.from_codes(arrs["scam_type"], categories=SCAM_TYPES),
        "stage": arrs["stage"].astype(np.int8),
        "is_scam": arrs["is_scam"].astype(np.int8),
        "is_in_report": arrs["is_in_report"].astype(np.int8),
    })
    return tx


def attach_day_split(cfg: Config, tx: pd.DataFrame) -> np.ndarray:
    codes = split_codes(cfg.n_days, cfg.split_fracs)
    return codes[tx["day"].to_numpy()]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic UPI scam-journey data.")
    ap.add_argument("--out-dir", default=None,
                    help="default: data/ (or data/smoke/ with --smoke)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--users", type=int, default=None)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    out_dir = args.out_dir or (SMOKE_DIR if args.smoke else "data")
    cfg = Config(seed=args.seed, out_dir=out_dir)
    if args.smoke:
        cfg = replace(cfg, smoke=True).resolved()
    if args.users is not None:
        cfg = replace(cfg, n_users=args.users)
    if args.days is not None:
        cfg = replace(cfg, n_days=args.days)

    run(cfg)


def run(cfg: Config) -> None:
    t0 = time.time()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    acc = build_accounts(cfg, make_rng(cfg.seed, "accounts"))
    us = generate_users(cfg, acc, make_rng(cfg.seed, "users"))

    mule_rng = make_rng(cfg.seed, "mules")
    n_mules = max(cfg.n_aggregators + 2, int(round(cfg.mule_rate * cfg.n_users)))
    mule_idx = mule_rng.permutation(cfg.n_users)[:n_mules]
    net = build_mule_network(cfg, acc, mule_idx, mule_rng)

    led = Ledger()
    add_normal(cfg, acc, us, make_rng(cfg.seed, "normal"), led)
    campaigns = generate_campaigns(cfg, acc, us, net, make_rng(cfg.seed, "campaigns"), led)

    tx = build_transactions(cfg, led, acc, len(campaigns))
    split_codes_arr = attach_day_split(cfg, tx)
    split_names = np.array(SPLITS)[split_codes_arr]
    tx.insert(3, "split", pd.Categorical(split_names, categories=SPLITS))

    acc.role[net.entry] = "mule"
    acc.role[net.mid] = "mule"

    campaigns_df = build_campaigns_df(campaigns)
    labels = build_labels(cfg, acc, net, campaigns, tx, split_codes_arr)
    edges = build_graph_edges(acc, tx, split_codes_arr, len(campaigns))

    report = validate(cfg, acc, tx, labels, edges, campaigns)

    if cfg.write_csv:
        tx.to_csv(out / "transactions.csv", index=False, chunksize=250_000)
        edges.to_csv(out / "graph_edges.csv", index=False)
        labels.to_csv(out / "labels.csv", index=False)

    print(f"[journey_generator] users={cfg.n_users} days={cfg.n_days} "
          f"transactions={len(tx):,} accounts={acc.n_accounts} campaigns={len(campaigns)}")
    print(f"[journey_generator] edges={len(edges):,} labels={len(labels):,} "
          f"scam_txns={int(tx['is_scam'].sum()):,}")
    print(f"[journey_generator] validation: {report}")
    print(f"[journey_generator] outputs written to {out.resolve()} ({time.time() - t0:.1f}s)")


def build_campaigns_df(campaigns: list) -> pd.DataFrame:
    return pd.DataFrame([{
        "campaign_id": f"C{c.idx:05d}", "scam_type": c.scam_type, "split": SPLITS[c.split],
        "victim_id": c.victim, "entry_mule_id": c.entry_mule, "aggregator_id": c.agg_used,
        "exit_id": c.exit_used, "last_ts": c.last_ts, "n_scam_txns": c.n_scam_txns,
        "reported": c.reported, "report_day": c.report_day,
        "report_delay_days": c.report_delay, "report_coverage": c.coverage,
    } for c in campaigns])


def build_labels(cfg: Config, acc: Accounts, net: MuleNet, campaigns: list,
                 tx: pd.DataFrame, split_codes_arr: np.ndarray) -> pd.DataFrame:
    n = acc.n_accounts
    counts = np.zeros((3, n), dtype=np.int64)
    snd = tx["sender_id"].to_numpy()
    rcv = tx["receiver_id"].to_numpy()
    for c in range(3):
        m = split_codes_arr == c
        if m.any():
            counts[c] += np.bincount(snd[m], minlength=n)
            counts[c] += np.bincount(rcv[m], minlength=n)
    acct_split = counts.argmax(axis=0)

    camp_of = {}
    mule_camps = {}
    agg_camps = {}
    for c in campaigns:
        camp_of[c.victim] = c
        for md in (c.entry_mule, *c.mids_used):
            mule_camps.setdefault(int(md), []).append(c)
        agg_camps.setdefault(c.agg_used, []).append(c)
        acct_split[c.victim] = c.split
        acct_split[c.entry_mule] = c.split
        for md in c.mids_used:
            acct_split[int(md)] = c.split
        acct_split[c.agg_used] = c.split

    for md, camps in mule_camps.items():
        vals, cnts = np.unique([c.split for c in camps], return_counts=True)
        acct_split[md] = int(vals[cnts.argmax()])

    role = acc.role
    gtl = np.array(["normal"] * n, dtype=object)
    gtl[role == "business"] = "business"
    gtl[role == "exit"] = "exit"
    gtl[role == "aggregator"] = "aggregator"
    gtl[role == "mule"] = "mule"
    gtl[role == "victim"] = "victim"

    is_reported = np.zeros(n, dtype=bool)
    report_day = np.full(n, np.nan)
    report_delay = np.full(n, np.nan)
    report_cov = np.full(n, np.nan)
    scam_type = np.full(n, "", dtype=object)
    campaign_id = np.full(n, "", dtype=object)

    for c in campaigns:
        cid = f"C{c.idx:05d}"
        targets = [c.victim, c.entry_mule, *c.mids_used, c.agg_used]
        for a in targets:
            scam_type[a] = c.scam_type
            campaign_id[a] = cid
        known = c.reported
        for a in targets:
            if known and not is_reported[a]:
                is_reported[a] = True
                report_day[a] = c.report_day
                report_delay[a] = c.report_delay
                report_cov[a] = c.coverage
            elif known:
                report_day[a] = min(report_day[a], c.report_day)
                report_delay[a] = min(report_delay[a], c.report_delay)
                report_cov[a] = max(report_cov[a], c.coverage)

    victim_ids = [c.victim for c in campaigns]
    acc.role[victim_ids] = "victim"
    gtl[victim_ids] = "victim"

    reported_lbl = gtl.copy()
    for a in np.where((gtl != "normal") & (gtl != "business") & (gtl != "exit"))[0]:
        if not is_reported[a]:
            reported_lbl[a] = "normal"

    edges_split = np.array(SPLITS)[acct_split]
    df = pd.DataFrame({
        "account_id": np.arange(n, dtype=np.int64),
        "vpa": acc.vpa,
        "role": role,
        "ground_truth_label": gtl,
        "reported_label": reported_lbl,
        "is_mule": (gtl == "mule").astype(np.int8),
        "is_victim": (gtl == "victim").astype(np.int8),
        "is_aggregator": (gtl == "aggregator").astype(np.int8),
        "is_business": (gtl == "business").astype(np.int8),
        "scam_type": pd.Series(scam_type).replace("", np.nan),
        "campaign_id": pd.Series(campaign_id).replace("", np.nan),
        "split": pd.Categorical(edges_split, categories=SPLITS),
        "join_day": acc.join_day.astype(np.int16),
        "is_reported": is_reported.astype(np.int8),
        "report_day": np.round(report_day, 2),
        "report_delay_days": np.round(report_delay, 2),
        "report_coverage": np.round(report_cov, 3),
    })
    return df


def build_graph_edges(acc: Accounts, tx: pd.DataFrame, split_codes_arr: np.ndarray,
                      n_campaigns: int) -> pd.DataFrame:
    sub = pd.DataFrame({
        "sender": tx["sender_id"].to_numpy(),
        "receiver": tx["receiver_id"].to_numpy(),
        "amount": tx["amount"].to_numpy(),
        "day": tx["day"].to_numpy(),
        "is_scam": tx["is_scam"].to_numpy(),
        "campaign": tx["campaign_id"].cat.codes.to_numpy(),
        "split": split_codes_arr,
    })
    g = sub.groupby(["sender", "receiver"], sort=False).agg(
        txn_count=("amount", "size"),
        total_amount=("amount", "sum"),
        first_day=("day", "min"),
        last_day=("day", "max"),
        n_scam=("is_scam", "sum"),
    ).reset_index()

    scam = sub[sub["campaign"] >= 0]
    if len(scam):
        cids = scam.groupby(["sender", "receiver"])["campaign"].first().rename("camp")
        g = g.merge(cids, on=["sender", "receiver"], how="left")
    else:
        g["camp"] = np.nan

    role = acc.role
    is_mule = (role[g["sender"].to_numpy()] == "mule") | (role[g["receiver"].to_numpy()] == "mule")
    g["edge_id"] = np.arange(len(g), dtype=np.int64)
    g["src_vpa"] = pd.Categorical.from_codes(g["sender"].to_numpy(), categories=acc.vpa)
    g["dst_vpa"] = pd.Categorical.from_codes(g["receiver"].to_numpy(), categories=acc.vpa)
    fd = g["first_day"].to_numpy().astype(np.int64)
    ld = g["last_day"].to_numpy().astype(np.int64)
    g["first_ts"] = np.datetime64("2024-01-01") + (fd * 86400).astype("timedelta64[s]")
    g["last_ts"] = np.datetime64("2024-01-01") + (ld * 86400).astype("timedelta64[s]")
    g["total_amount"] = np.round(g["total_amount"], 2)
    g["is_mule_edge"] = is_mule.astype(np.int8)
    camp = g["camp"].to_numpy()
    camp_codes = np.where(np.isnan(camp), -1, camp).astype(np.int64)
    camp_cats = [f"C{i:05d}" for i in range(n_campaigns)] if n_campaigns else []
    g["campaign_id"] = pd.Categorical.from_codes(camp_codes, categories=camp_cats)

    scnt = sub.groupby(["sender", "receiver", "split"], sort=False).size().rename("ns").reset_index()
    best = scnt.loc[scnt.groupby(["sender", "receiver"])["ns"].idxmax(), ["sender", "receiver", "split"]]
    g = g.merge(best, on=["sender", "receiver"], how="left")
    g["split"] = pd.Categorical(np.array(SPLITS)[g["split"].to_numpy().astype(np.int64)], categories=SPLITS)

    return g[[
        "edge_id", "src_vpa", "dst_vpa", "txn_count", "total_amount", "first_ts", "last_ts",
        "n_scam", "is_mule_edge", "campaign_id", "split",
    ]]


def validate(cfg: Config, acc: Accounts, tx: pd.DataFrame, labels: pd.DataFrame,
             edges: pd.DataFrame, campaigns: list) -> dict:
    report = {}
    day_to_split = split_codes(cfg.n_days, cfg.split_fracs)
    camp_split = np.full(max(1, len(campaigns)), -1, dtype=np.int64)
    for c in campaigns:
        camp_split[c.idx] = c.split
    camp_codes = tx["campaign_id"].cat.codes.to_numpy()
    scam = tx[camp_codes >= 0]
    scam_codes = camp_codes[camp_codes >= 0]
    camp_days = scam.groupby(scam_codes)["day"].agg(["min", "max"])
    violations = 0
    for cid, row in camp_days.iterrows():
        s = day_to_split[int(row["min"])]
        if s != day_to_split[int(row["max"])] or s != camp_split[int(cid)]:
            violations += 1
    report["campaign_split_violations"] = int(violations)

    bad_ts = int((scam["ts"].isna()).sum())
    report["scam_ts_ok"] = bool(bad_ts == 0)
    report["labels_rows"] = int(len(labels))
    report["mules"] = int(labels["is_mule"].sum())
    report["victims"] = int(labels["is_victim"].sum())
    report["reported_labels"] = int((labels["reported_label"] != "normal").sum())
    report["split_sizes"] = {s: int((tx["split"] == s).sum()) for s in SPLITS}
    report["txn_count"] = int(len(tx))
    report["edge_count"] = int(len(edges))
    report["mule_edges"] = int(edges["is_mule_edge"].sum())

    maxid = int(max(tx["sender_id"].max(), tx["receiver_id"].max()))
    report["ref_integrity"] = bool(maxid < acc.n_accounts)
    return report


if __name__ == "__main__":
    main()
