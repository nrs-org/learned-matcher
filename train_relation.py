"""Relation head for track pairs the identity model calls `related`:
  kind      : cover / instrumental / live / remix / edit / arrangement / other
  direction : which side is derived (only where the label has one)

Direction is trained with swap augmentation (every example also appears with
sides exchanged, signed features negated, label flipped) and predicted as
(f(a,b) + 1 - f(b,a)) / 2, so it is antisymmetric by construction.

    uv run python train_relation.py --name rel-v1
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from features import ROOT

OUT = ROOT / "data/learned-matcher"
META = {"tier", "type", "label", "kind", "derived", "split", "entry_a", "entry_b", "why", "item_id", "weight", "stratum", "batch"}
KIND_MAP = {"cover": "cover", "instrumental": "instrumental", "live": "live", "remix": "remix", "edit": "edit",
            "arrangement": "arrangement", "other": "other", "alt_version": "other"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="rel-v1")
    ap.add_argument("--drop", action="append", default=[], help="ablation: drop features containing this substring")
    args = ap.parse_args()
    df = pd.read_parquet(OUT / "train_features.parquet")
    df = df[(df.type == "track") & df.label.isin(["related", "sibling"]) & df.kind.notna()].copy()
    df["kind"] = df.kind.map(KIND_MAP)
    df = df[df.kind.notna()]
    keep = lambda c: not any(d in c for d in args.drop)
    sym = [c for c in df.columns if c not in META and not c.startswith("s_") and keep(c)]
    signed = [c for c in df.columns if c.startswith("s_") and keep(c)]
    kinds = sorted(df.kind.unique())
    print(df.groupby(["kind", "split"]).size().unstack())

    tr, dv = df.split == "train", df.split == "dev"
    # kind: symmetric + |signed| (which side carries the marker doesn't matter)
    def kind_X(d):
        X = d[sym].copy()
        for c in signed:
            X["abs_" + c] = d[c].abs()
        return X
    yk = df.kind.map(kinds.index)
    counts = yk[tr].value_counts()
    wk = yk.map(lambda k: 1 / np.sqrt(counts.get(k, 1))).to_numpy()
    params = dict(objective="multiclass", num_class=len(kinds), learning_rate=0.05, num_leaves=31, min_data_in_leaf=10,
                  feature_fraction=0.7, verbose=-1, num_threads=12)
    km = lgb.train(params, lgb.Dataset(kind_X(df[tr]), yk[tr], weight=wk[tr.to_numpy()]), 500,
                   valid_sets=[lgb.Dataset(kind_X(df[dv]), yk[dv])], callbacks=[lgb.early_stopping(40, verbose=False)])
    pk = km.predict(kind_X(df[dv]), num_iteration=km.best_iteration).argmax(1)
    print("\nKIND (dev)\n" + classification_report(yk[dv], pk, labels=range(len(kinds)), target_names=kinds, zero_division=0))

    # direction
    d = df[df.derived.isin(["left", "right"])]
    def dir_X(d, flip=False):
        X = d[sym + signed].copy()
        if flip:
            X[signed] = -X[signed]
        return X
    def dir_y(d, flip=False):
        y = (d.derived == "left").astype(int).to_numpy()
        return 1 - y if flip else y
    dtr, ddv = d[d.split == "train"], d[d.split == "dev"]
    Xtr = pd.concat([dir_X(dtr), dir_X(dtr, True)])
    ytr = np.concatenate([dir_y(dtr), dir_y(dtr, True)])
    dm = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=10, verbose=-1, num_threads=12),
                   lgb.Dataset(Xtr, ytr), 300)
    p = (dm.predict(dir_X(ddv)) + 1 - dm.predict(dir_X(ddv, True))) / 2
    acc = ((p > 0.5).astype(int) == dir_y(ddv)).mean()
    conf = np.abs(p - 0.5) > 0.3
    print(f"DIRECTION (dev, n={len(ddv)}): accuracy {acc:.3f}; on confident (|p-0.5|>0.3, {conf.mean():.0%} of pairs): "
          f"{((p[conf] > 0.5).astype(int) == dir_y(ddv)[conf]).mean():.3f}")
    print(pd.DataFrame({"kind": ddv.kind, "ok": (p > 0.5).astype(int) == dir_y(ddv)}).groupby("kind").ok.agg(["mean", "size"]))

    # structure: left derived / right derived / sibling (neither derives from
    # the other). Swap-augmented: flipping sides negates signed features and
    # swaps left/right; sibling stays sibling.
    full = pd.read_parquet(OUT / "train_features.parquet")
    sd = full[(full.type == "track") & ((full.label == "sibling") | ((full.label == "related") & full.derived.isin(["left", "right"])))]
    def st_y(d, flip=False):
        y = np.where(d.label == "sibling", 2, np.where(d.derived == "left", 0, 1))
        return np.where(y == 2, 2, 1 - y) if flip else y
    s_tr, s_dv = sd[sd.split == "train"], sd[sd.split == "dev"]
    counts = np.bincount(st_y(s_tr), minlength=3)
    ytr_s = np.concatenate([st_y(s_tr), st_y(s_tr, True)])
    wtr_s = 1 / np.sqrt(counts[ytr_s])
    sm = lgb.train(dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=31, min_data_in_leaf=10,
                        feature_fraction=0.7, verbose=-1, num_threads=12),
                   lgb.Dataset(pd.concat([dir_X(s_tr), dir_X(s_tr, True)]), ytr_s, weight=wtr_s), 400)
    p1, p2 = sm.predict(dir_X(s_dv)), sm.predict(dir_X(s_dv, True))
    p_sib = (p1[:, 2] + p2[:, 2]) / 2
    ys = st_y(s_dv) == 2
    print(f"STRUCTURE (dev, n={len(s_dv)}, siblings {ys.sum()}): sibling P/R at 0.5 = "
          f"{(ys & (p_sib > 0.5)).sum() / max((p_sib > 0.5).sum(), 1):.3f} / {(ys & (p_sib > 0.5)).sum() / max(ys.sum(), 1):.3f}; "
          f"by tier {pd.DataFrame({'tier': s_dv.tier, 'ok': (p_sib > 0.5) == ys}).groupby('tier').ok.mean().round(3).to_dict()}")

    out = OUT / "models" / args.name
    out.mkdir(parents=True, exist_ok=True)
    km.save_model(str(out / "kind.txt"), num_iteration=km.best_iteration)
    dm.save_model(str(out / "direction.txt"))
    sm.save_model(str(out / "structure.txt"))
    (out / "kinds.json").write_text(json.dumps(kinds))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
