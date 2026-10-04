"""Re-predict the gold feature table (all gold batches) with a saved model.

    uv run python predict_gold.py --model v9
"""
import argparse
import json

import lightgbm as lgb
import pandas as pd

from features import ROOT
from policy import class_names, merge_guard, verdicts
from structure import structure_probs

OUT = ROOT / "data/learned-matcher"

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--structure", default="rel-v6", help="relation-head dir with structure.txt ('' to disable)")
args = ap.parse_args()
mdir = OUT / "models" / args.model
m = lgb.Booster(model_file=str(mdir / "model.txt"))
th = json.loads((mdir / "thresholds.json").read_text())
gold = pd.read_parquet(OUT / "gold_features.parquet")
P = m.predict(gold[m.feature_name()])
gp = gold[["item_id"]].copy()
gp[[f"p_{c}" for c in class_names(P.shape[1])]] = P
gp["verdict"] = verdicts(gold["type"], P, th, guard=merge_guard(gold),
                         structure=structure_probs(gold, gold["type"], args.structure))
gp.to_csv(mdir / "gold_preds.csv", index=False)
print(f"{len(gp)} gold predictions -> {mdir / 'gold_preds.csv'}")
