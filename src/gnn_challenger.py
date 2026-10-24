"""Phase 7: the PyG GraphSAGE challenger to the XGBoost champion.

The Maya PDF (p. 15) floats a GNN challenger that reasons over the *graph* of
accounts rather than per-transaction features. This script builds that
challenger: a 2-layer GraphSAGE node classifier that predicts whether an
account is a mule (`labels.is_mule`), then prints the verdict table against the
XGBoost champion with a verdict derived from the measured AUROC and latency.

Run:
    python3 src/gnn_challenger.py --smoke       # 3 epochs, fast sanity check
    python3 src/gnn_challenger.py               # full training + table
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.jit.script.*")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected

from sklearn.metrics import roc_auc_score

from risk_common import LATENCY_SLO_MS  # noqa: E402

CHAMPION_LATENCY_MS = LATENCY_SLO_MS  # per-payment SLO from the spec
GNN_USEFUL_AUROC = 0.70  # below this the challenger has not learned mule-ness


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.lin = torch.nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index).relu()
        return self.lin(x).squeeze(-1)


EPOCH = pd.Timestamp("2024-01-01")  # day 0 of the generator (transactions.day)


def split_start_days(txns_path: str) -> dict:
    """First day of each split, from transactions.csv (contiguous day blocks)."""
    t = pd.read_csv(txns_path, usecols=["day", "split"])
    return t.groupby("split")["day"].min().astype(int).to_dict()


def snapshot(src, dst, n_nodes: int, y, masks: dict) -> Data:
    """One graph snapshot: degree features + undirected edges from the given edge set."""
    # Directed degrees -> structural node features (no label leakage).
    in_deg = np.bincount(dst, minlength=n_nodes).astype(np.float32)
    out_deg = np.bincount(src, minlength=n_nodes).astype(np.float32)
    x = np.stack([np.log1p(in_deg), np.log1p(out_deg), np.log1p(in_deg + out_deg)], axis=1)
    edge_index = to_undirected(torch.tensor(np.stack([src, dst]), dtype=torch.long))
    return Data(x=torch.tensor(x, dtype=torch.float32), edge_index=edge_index, y=y, **masks)


def build_graph(labels_path: str, edges_path: str, txns_path: str, smoke: bool) -> tuple[dict, dict]:
    """Nodes = accounts (labels.csv), edges = aggregated txns (graph_edges.csv).

    Node features are leakage-free graph statistics (degree); the label `role`
    and its derivatives (`is_mule`, `is_aggregator`, ...) are deliberately
    excluded so the model must *learn* mule-ness from neighbourhood structure.

    Past-only: returns three time-sliced snapshots keyed by split. An edge exists
    in a snapshot only if its first txn (`first_ts`) is before the cutoff:
        train -> before the first val day   (fit)
        val   -> before the first test day  (early stopping)
        test  -> full history               (final score)
    so training never sees edges from the val/test periods.
    """
    labels = pd.read_csv(labels_path)
    edges = pd.read_csv(edges_path)
    starts = split_start_days(txns_path)

    vpa_to_idx = {vpa: i for i, vpa in enumerate(labels["vpa"])}
    n_nodes = len(labels)

    src = edges["src_vpa"].map(vpa_to_idx)
    dst = edges["dst_vpa"].map(vpa_to_idx)
    valid = src.notna() & dst.notna()
    dropped = int((~valid).sum())
    src = src[valid].astype(np.int64).to_numpy()
    dst = dst[valid].astype(np.int64).to_numpy()
    first_day = (pd.to_datetime(edges.loc[valid, "first_ts"]) - EPOCH).dt.days.to_numpy()

    if smoke:
        keep = np.arange(min(len(src), 200_000))
        src, dst, first_day = src[keep], dst[keep], first_day[keep]

    y = torch.tensor(labels["is_mule"].to_numpy(), dtype=torch.float32)
    split = labels["split"].to_numpy()
    masks = {f"{s}_mask": torch.tensor(split == s) for s in ("train", "val", "test")}

    full = max(int(first_day.max()) + 1, starts["test"])
    cutoffs = {"train": starts["val"], "val": starts["test"], "test": full}
    snaps = {}
    for name, cutoff in cutoffs.items():
        keep = first_day < cutoff
        snaps[name] = snapshot(src[keep], dst[keep], n_nodes, y, masks)

    is_mule = labels["is_mule"].to_numpy() == 1
    stats = {
        "n_nodes": n_nodes,
        "n_edges": {k: int(v.edge_index.size(1)) for k, v in snaps.items()},
        "cutoff_day": cutoffs,
        "n_dropped_edges": dropped,
        "n_mules": int(is_mule.sum()),
        **{f"n_{s}": int((split == s).sum()) for s in ("train", "val", "test")},
        **{f"n_{s}_mules": int((is_mule & (split == s)).sum()) for s in ("train", "val", "test")},
    }
    return snaps, stats


def safe_auroc(y_true, y_score) -> float | None:
    """AUROC, or None if the split has only one class (degenerate)."""
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def train(snaps: dict, epochs: int, hidden: int, lr: float, seed: int) -> tuple[GraphSAGE, float, float]:
    """Fit on the train snapshot, early-stop on the val snapshot, score the test snapshot."""
    tr, va, te = snaps["train"], snaps["val"], snaps["test"]
    torch.manual_seed(seed)
    model = GraphSAGE(tr.x.size(1), hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_pos = int(tr.y[tr.train_mask].sum())
    n_neg = int(tr.train_mask.sum()) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val, best_state, patience = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(tr.x, tr.edge_index)
        loss = criterion(out[tr.train_mask], tr.y[tr.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(va.x, va.edge_index))
        val_auc = safe_auroc(
            va.y[va.val_mask].numpy(), prob[va.val_mask].numpy()
        )
        if val_auc is not None and val_auc > best_val:
            best_val = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 25:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(te.x, te.edge_index))
    test_auc = safe_auroc(te.y[te.test_mask].numpy(), prob[te.test_mask].numpy())
    return model, (best_val if best_val >= 0 else None), test_auc


def measure_latency_ms(model: GraphSAGE, data: Data, reps: int = 5) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(2):  # warmup
            model(data.x, data.edge_index)
        t0 = time.perf_counter()
        for _ in range(reps):
            model(data.x, data.edge_index)
        return (time.perf_counter() - t0) / reps * 1000.0


def print_table(champion_auroc: float, gnn_auroc: float | None, gnn_latency_ms: float) -> None:
    gnn = f"{gnn_auroc:.4f}" if gnn_auroc is not None else "n/a (1 class)"
    print()
    print("Model                 | Predicts            | AUROC   | Latency")
    print("----------------------|---------------------|---------|-------------")
    print(f"XGBoost (champion)    | txn-level is_scam   | {champion_auroc:.4f} | ~{CHAMPION_LATENCY_MS:.0f} ms (hot path)")
    print(f"GNN GraphSAGE         | account is_mule     | {gnn} | {gnn_latency_ms:.1f} ms (full graph fwd)")
    print()
    print("Note: different targets. The champion scores each *transaction* in the hot")
    print("path, while the GNN labels the *account* as a mule by aggregating 2-hop")
    print("neighbourhood structure (victim -> entry mule -> mid mule -> aggregator")
    print("chains) that the per-transaction features only approximate via d_R.")
    print()
    for line in verdict_lines(gnn_auroc, gnn_latency_ms):
        print(line)


def verdict_lines(gnn_auroc: float | None, gnn_latency_ms: float) -> list[str]:
    """Verdict derived from the measured GNN numbers (never hardcoded).

    The two AUROCs score different targets, so the GNN is judged on whether it
    learned a usable account-level mule signal and whether it fits the SLO --
    not on beating the champion's txn-level number.
    """
    if gnn_auroc is None:
        quality = "the GNN could not be scored (test split has a single class)."
    elif gnn_auroc < GNN_USEFUL_AUROC:
        quality = (f"the GNN did not learn a usable mule signal (test AUROC "
                   f"{gnn_auroc:.4f} < {GNN_USEFUL_AUROC:.2f}).")
    else:
        quality = (f"the GNN flags mule accounts well (test AUROC {gnn_auroc:.4f} on "
                   f"is_mule), a target the champion does not score, so it complements "
                   f"rather than replaces XGBoost.")
    slo = CHAMPION_LATENCY_MS
    if gnn_latency_ms > slo:
        latency = (f"Its full-graph forward pass takes {gnn_latency_ms:.1f} ms, over the "
                   f"{slo:.0f} ms per-payment SLO.")
    else:
        latency = (f"Its full-graph forward pass ({gnn_latency_ms:.1f} ms) fits the "
                   f"{slo:.0f} ms SLO on this graph, but needs the whole graph resident "
                   f"at inference.")
    return [
        f"Verdict: {quality}",
        latency,
        "It does not score individual payments, so XGBoost stays in the per-payment hot path.",
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PyG GraphSAGE challenger vs XGBoost champion.")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--edges", default="data/graph_edges.csv")
    ap.add_argument("--txns", default="data/transactions.csv",
                    help="read for the split start days (snapshot cutoffs)")
    ap.add_argument("--meta", default="models/model_meta.json")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    if args.smoke:
        args.epochs = 3

    snaps, stats = build_graph(args.labels, args.edges, args.txns, args.smoke)
    print(f"[gnn] nodes={stats['n_nodes']} dropped_edges={stats['n_dropped_edges']} "
          f"snapshot edges={stats['n_edges']} (first_ts < cutoff day {stats['cutoff_day']})")
    print(f"[gnn] accounts train/val/test={stats['n_train']}/{stats['n_val']}/{stats['n_test']} "
          f"mules={stats['n_train_mules']}/{stats['n_val_mules']}/{stats['n_test_mules']}")
    if min(stats["n_val_mules"], stats["n_test_mules"]) < 10:
        print(f"[gnn] WARNING: only {stats['n_val_mules']} val / {stats['n_test_mules']} test mules; "
              "val/test AUROC are high-variance and should not be over-read.")

    model, val_auc, test_auc = train(snaps, args.epochs, args.hidden, args.lr, args.seed)
    latency = measure_latency_ms(model, snaps["test"])

    assert val_auc is None or np.isfinite(val_auc), "non-finite val AUROC"
    print(f"[gnn] val_auroc={val_auc if val_auc is None else round(val_auc, 4)} "
          f"test_auroc={test_auc if test_auc is None else round(test_auc, 4)} "
          f"fwd_latency={latency:.1f}ms")

    champion = json.loads(Path(args.meta).read_text())["test_auroc"]
    print_table(champion, test_auc, latency)

    if args.smoke:
        print("[gnn] SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
