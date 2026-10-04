"""Train the pair classifier (LightGBM, 3 classes: same / related / unrelated)
and report dev metrics next to the Rhai baseline.

    uv run python train.py [--name v1] [--drop-encoder]
Writes data/learned-matcher/models/<name>/{model.txt,dev_report.json,gold_preds.csv}
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from features import ROOT

OUT = ROOT / "data/learned-matcher"
from policy import CLASSES, merge_guard, verdicts
META = {"tier", "type", "label", "kind", "derived", "split", "entry_a", "entry_b", "why", "item_id", "weight", "stratum", "batch"}
VERDICT_CLASS = {"MERGE": "same", "RELATE": "related", "DISTINCT": "unrelated"}


def recall_at_precision(y, s, target=0.99):
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    ok = np.where(prec >= target)[0]
    return (tp[ok].max() / y.sum()) if len(ok) and y.sum() else 0.0


def rhai_lookup():
    cand = pd.read_csv(ROOT / "data/eval/candidates-rhai-2026-10-03.csv", low_memory=False,
                       usecols=["verdict", "entry_a", "entry_b"])
    cand = cand[cand.verdict.isin(VERDICT_CLASS)]
    a, b = cand.entry_a.to_numpy(), cand.entry_b.to_numpy()
    keys = [f"{min(x, y)}-{max(x, y)}" for x, y in zip(a, b)]
    return dict(zip(keys, cand.verdict.map(VERDICT_CLASS)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v1")
    ap.add_argument("--drop-encoder", action="store_true", help="ablation: no encoder features")
    ap.add_argument("--drop", action="append", default=[], help="ablation: drop features containing this substring")
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--min-data", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--ff", type=float, default=0.7)
    ap.add_argument("--merge-precision", type=float, default=0.98)
    ap.add_argument("--merge-precision-artist", type=float, default=0.99,
                    help="artist dev negatives are easier than real artist candidates; aim higher")
    args = ap.parse_args()

    df = pd.read_parquet(OUT / "train_features.parquet")
    gold = pd.read_parquet(OUT / "gold_features.parquet")
    feats = [c for c in df.columns if c not in META and not c.startswith("s_")]
    if args.drop_encoder:
        feats = [c for c in feats if not (c.startswith("enc_") or c.startswith("artist_enc") or c[0] in "dp" and c[1:].isdigit())]
    feats = [c for c in feats if not any(d in c for d in args.drop)]
    y = df["label"].map(CLASSES.index).to_numpy()

    # Balance classes per type (sqrt inverse frequency) so 110k easy negatives
    # don't drown 3.7k positives; probabilities are re-calibrated on dev.
    counts = df.groupby(["type", "label"]).size()
    w = df.apply(lambda r: 1.0 / np.sqrt(counts[(r["type"], r["label"])]), axis=1).to_numpy()
    w = w / w.mean()

    tr, dv = (df["split"] == "train").to_numpy(), (df["split"] == "dev").to_numpy()
    params = dict(objective="multiclass", num_class=len(CLASSES), learning_rate=args.lr, num_leaves=args.leaves,
                  min_data_in_leaf=args.min_data, feature_fraction=args.ff, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=12)
    dtr = lgb.Dataset(df.loc[tr, feats], y[tr], weight=w[tr])
    ddv = lgb.Dataset(df.loc[dv, feats], y[dv], weight=w[dv], reference=dtr)
    model = lgb.train(params, dtr, args.rounds, valid_sets=[ddv], callbacks=[lgb.early_stopping(50, verbose=False)])
    print(f"best iteration {model.best_iteration}; dev weighted multi_logloss {model.best_score['valid_0']['multi_logloss']:.5f}")

    P = model.predict(df.loc[dv, feats], num_iteration=model.best_iteration)
    dev = df.loc[dv, ["type", "tier", "label", "entry_a", "entry_b", "why"]].copy()
    dev[[f"p_{c}" for c in CLASSES]] = P
    dev["pred"] = [CLASSES[i] for i in P.argmax(1)]
    rh = rhai_lookup()
    dev["rhai"] = [rh.get(f"{min(a, b)}-{max(a, b)}") if a != b else None for a, b in zip(dev.entry_a, dev.entry_b)]

    report = {"name": args.name, "features": len(feats), "best_iteration": model.best_iteration, "dev": {}}
    print(f"\n{'type':14s} {'n':>7s} {'same':>6s} {'AP_same':>8s} {'R@P99':>6s} {'AP_rel':>7s} {'acc':>6s} | rhai on mb tier: {'P_same':>6s} {'R_same':>6s}  model same P/R on same pairs")
    for typ, part in dev.groupby("type"):
        ys = (part.label == "same").to_numpy()
        yr = (part.label == "related").to_numpy()
        r = {"n": len(part), "n_same": int(ys.sum()), "n_related": int(yr.sum())}
        r["ap_same"] = average_precision_score(ys, part.p_same) if ys.any() else np.nan
        r["recall_at_p99_same"] = recall_at_precision(ys, part.p_same.to_numpy())
        r["ap_related"] = average_precision_score(yr, part.p_related) if yr.any() else np.nan
        r["accuracy"] = float((part.pred == part.label).mean())
        # head-to-head on real candidate pairs that Rhai also scored
        mb = part[part.rhai.notna() & (part.tier == "mb")]
        if len(mb):
            for who in ("rhai", "pred"):
                ps, gs = mb[who] == "same", mb.label == "same"
                r[f"{who}_same_precision_mb"] = float((ps & gs).sum() / max(ps.sum(), 1))
                r[f"{who}_same_recall_mb"] = float((ps & gs).sum() / max(gs.sum(), 1))
                pr, gr = mb[who] == "related", mb.label == "related"
                r[f"{who}_related_precision_mb"] = float((pr & gr).sum() / max(pr.sum(), 1))
                r[f"{who}_related_recall_mb"] = float((pr & gr).sum() / max(gr.sum(), 1))
                r[f"{who}_accuracy_mb"] = float((mb[who] == mb.label).mean())
            r["n_mb"] = len(mb)
        report["dev"][typ] = r
        print(f"{typ:14s} {r['n']:7d} {r['n_same']:6d} {r['ap_same']:8.3f} {r['recall_at_p99_same']:6.3f} {r['ap_related']:7.3f} {r['accuracy']:6.3f}")
        if len(mb):
            print(f"{'':14s} on {len(mb)} real candidate pairs: accuracy rhai {r['rhai_accuracy_mb']:.3f} vs model {r['pred_accuracy_mb']:.3f};"
                  f" related P/R rhai {r['rhai_related_precision_mb']:.2f}/{r['rhai_related_recall_mb']:.2f}"
                  f" model {r['pred_related_precision_mb']:.2f}/{r['pred_related_recall_mb']:.2f};"
                  f" same P/R rhai {r['rhai_same_precision_mb']:.2f}/{r['rhai_same_recall_mb']:.2f}"
                  f" model {r['pred_same_precision_mb']:.2f}/{r['pred_same_recall_mb']:.2f}")
    # per tier
    print()
    for (typ, tier), part in dev.groupby(["type", "tier"]):
        print(f"  {typ:14s} {tier:10s} n={len(part):6d} acc={float((part.pred == part.label).mean()):.3f}"
              f"  confusion {pd.crosstab(part.label, part.pred).to_dict()}")

    imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("\ntop features by gain:\n" + imp.head(25).round(0).to_string())

    out = OUT / "models" / args.name
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / "model.txt"), num_iteration=model.best_iteration)
    dev.to_parquet(out / "dev_preds.parquet")
    Path(out / "dev_report.json").write_text(json.dumps(report, indent=1, default=float))
    # Policy (policy.verdicts): MERGE only above a per-type threshold chosen
    # for `--merge-precision` on dev rows whose labels are trustworthy for
    # that type -- every tier for tracks; for artists / releases / groups the
    # MB "distinct" tier is too noisy (homonyms, aliases), so only split
    # halves, half-view negatives and Jev labels. Silver-set pairs are
    # excluded so the silver test stays untouched.
    silver = set()
    sp = ROOT / "data/eval/gold/silver-jev-v1.items.jsonl"
    if sp.exists():
        for l in sp.read_text().splitlines():
            it = json.loads(l)
            silver.add((min(it["a"]["entry_id"], it["b"]["entry_id"]), max(it["a"]["entry_id"], it["b"]["entry_id"])))
    thresholds = {}
    for typ, part in dev.groupby("type"):
        if typ == "artist":
            # split-half artist pairs are too easy to set a cut on; Jev-labeled
            # real candidates are the closest to what the policy will see
            part = part[part.tier == "jev"]
        elif typ != "track":
            part = part[part.tier.isin(["probe", "probe_neg", "jev"])]
        part = part[[(min(a, b), max(a, b)) not in silver for a, b in zip(part.entry_a, part.entry_b)]]
        ys = (part.label == "same").to_numpy()
        s = part.p_same.to_numpy()
        order = np.argsort(-s)
        tp = np.cumsum(ys[order])
        prec = tp / np.arange(1, len(s) + 1)
        target = args.merge_precision_artist if typ == "artist" else args.merge_precision
        ok = np.where(prec >= target)[0]
        thresholds[typ] = float(s[order][ok.max()]) if len(ok) else 1.0
    print("merge thresholds:", {k: round(v, 3) for k, v in thresholds.items()})
    Path(out / "thresholds.json").write_text(json.dumps(thresholds, indent=1))

    G = model.predict(gold[feats], num_iteration=model.best_iteration)
    gp = gold[["item_id"]].copy()
    gp[[f"p_{c}" for c in CLASSES]] = G
    gp["verdict"] = verdicts(gold["type"], G, thresholds, guard=merge_guard(gold))
    gp.to_csv(out / "gold_preds.csv", index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
