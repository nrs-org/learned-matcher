"""Score predictions against the human gold set.

Gold items come from `sample_gold.py` (each carries its stratum weight
N_h / n_h); labels are exported from the labeling page
(`data/eval/gold/<batch>.labels.jsonl`). Every estimate is a weighted
(Horvitz-Thompson ratio) estimate over the whole candidate population, with a
stratified bootstrap 90% interval.

Predictions: a CSV with `item_id` and either `verdict` (MERGE/RELATE/DISTINCT)
or `p_same` (+ optional `p_related`). `--rhai` scores the Rhai verdict stored
on each item instead.

    uv run python evaluate.py --batch human-v1 --rhai
    uv run python evaluate.py --batch human-v1 --pred preds.csv
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
import pandas as pd

GOLD_DIR = Path(__file__).resolve().parents[2] / "data/eval/gold"

# Human label -> coarse class used by every metric.
CLASS = {
    "same": "same",
    "a_from_b": "related", "b_from_a": "related", "related": "related",
    # siblings (two versions of one song): no direct edge, each links to the
    # common original (user, 2026-10-03)
    "siblings": "sibling",
    "different": "unrelated",
}
VERDICT_CLASS = {"MERGE": "same", "RELATE": "related", "SIBLING": "sibling", "DISTINCT": "unrelated", "DEFER": "defer"}


def load_gold(batches, min_label_conf=0.0):
    rows = []
    for batch in batches:
        items = [json.loads(l) for l in (GOLD_DIR / f"{batch}.items.jsonl").read_text().splitlines()]
        labels_path = GOLD_DIR / f"{batch}.labels.jsonl"
        labels = {}
        if labels_path.exists():
            for l in labels_path.read_text().splitlines():
                d = json.loads(l)
                labels[d["item_id"]] = d
        for it in items:
            lab = labels.get(it["item_id"], {})
            ident = lab.get("identity")
            if lab.get("confidence") is not None and lab["confidence"] < min_label_conf:
                ident = None  # judge not confident enough: leave unlabeled
            rows.append({
                "item_id": it["item_id"], "type": it["type"], "stratum": it["stratum"],
                "weight": it["weight"], "rhai_verdict": it["rhai"]["verdict"],
                "label": ident, "kind": lab.get("kind"),
                "gold": CLASS.get(ident) if ident else None,
            })
    return pd.DataFrame(rows)


def ratio(num_mask, den_mask, w):
    den = w[den_mask].sum()
    return np.nan if den == 0 else w[num_mask & den_mask].sum() / den


def point_metrics(df, pred_col):
    w = df["weight"].to_numpy()
    g = df["gold"].to_numpy()
    p = df[pred_col].to_numpy()
    out = {}
    for cls in ("same", "related", "sibling"):
        out[f"{cls}_precision"] = ratio(g == cls, p == cls, w)
        out[f"{cls}_recall"] = ratio(p == cls, g == cls, w)
    # "Direct link" = same or related (siblings get no direct edge).
    link_g, link_p = np.isin(g, ["same", "related"]), np.isin(p, ["same", "related"])
    out["link_precision"] = ratio(link_g, link_p, w)
    out["link_recall"] = ratio(link_p, link_g, w)
    decided = p != "defer"
    out["accuracy"] = ratio(g == p, decided, w)
    out["deferred"] = ratio(~decided, np.ones_like(g, dtype=bool), w)
    return out


def bootstrap(df, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("stratum")]
    samples = defaultdict(list)
    for _ in range(n):
        parts = [g.iloc[rng.integers(0, len(g), len(g))] for g in groups]
        for k, v in fn(pd.concat(parts)).items():
            samples[k].append(v)
    return {k: (np.nanpercentile(v, 5), np.nanpercentile(v, 95)) for k, v in samples.items()}


def recall_at_precision(df, score_col, target=0.99):
    """Highest weighted `same` recall over thresholds whose weighted precision >= target."""
    d = df.sort_values(score_col, ascending=False)
    w, pos = d["weight"].to_numpy(), (d["gold"] == "same").to_numpy()
    tp, fp = np.cumsum(w * pos), np.cumsum(w * ~pos)
    total = (w * pos).sum()
    if total == 0:
        return np.nan, np.nan
    prec = tp / np.maximum(tp + fp, 1e-12)
    ok = np.where(prec >= target)[0]
    if len(ok) == 0:
        return 0.0, np.nan
    i = ok[np.argmax(tp[ok])]
    return tp[i] / total, d[score_col].iloc[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", action="append", required=True)
    ap.add_argument("--rhai", action="store_true")
    ap.add_argument("--pred")
    ap.add_argument("--out")
    ap.add_argument("--min-label-conf", type=float, default=0.0,
                    help="silver sets: ignore judge labels below this confidence")
    args = ap.parse_args()

    df = load_gold(args.batch, args.min_label_conf)
    total = len(df)
    labeled = df[df["gold"].notna()].copy()
    unsure = int((df["label"] == "unsure").sum())
    print(f"items {total}, labeled {len(labeled)}, can't-tell {unsure}, unlabeled {int(df['label'].isna().sum())}")
    if labeled.empty:
        return

    if args.rhai:
        labeled["pred"] = labeled["rhai_verdict"].map(VERDICT_CLASS)
        name = "rhai"
    else:
        pred = pd.read_csv(args.pred)
        labeled = labeled.merge(pred, on="item_id", how="left")
        if "verdict" in labeled:
            labeled["pred"] = labeled["verdict"].map(VERDICT_CLASS)
        name = Path(args.pred).stem

    report = {"name": name, "batches": args.batch, "labeled": len(labeled), "by_type": {}}
    for typ, part in [("all", labeled)] + list(labeled.groupby("type")):
        entry = {"n": len(part), "gold_counts": part["gold"].value_counts().to_dict()}
        if "pred" in part:
            pm = point_metrics(part, "pred")
            ci = bootstrap(part, lambda d: point_metrics(d, "pred"), n=1000)
            entry["point"] = {k: (v, *ci.get(k, (np.nan, np.nan))) for k, v in pm.items()}
            entry["confusion"] = pd.crosstab(part["gold"], part["pred"], values=part["weight"], aggfunc="sum").fillna(0).round(0).to_dict()
        if "p_same" in part:
            r, thr = recall_at_precision(part, "p_same")
            entry["same_recall_at_p99"] = r
            entry["threshold_at_p99"] = thr
        report["by_type"][typ] = entry

    for typ, e in report["by_type"].items():
        print(f"\n== {typ}  (n={e['n']}, gold {e['gold_counts']})")
        for k, (v, lo, hi) in e.get("point", {}).items():
            print(f"  {k:18s} {v:6.3f}  [{lo:.3f}, {hi:.3f}]")
        if "same_recall_at_p99" in e:
            print(f"  same recall @ P>=0.99: {e['same_recall_at_p99']:.3f} (threshold {e['threshold_at_p99']})")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
