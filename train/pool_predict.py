"""Score the whole candidate pool with a trained model.

    uv run python pool_predict.py --model v6
Writes data/learned-matcher/models/<model>/pool_preds.parquet
(entry_a, entry_b, type, rhai, p_same, p_related, p_unrelated, verdict).
"""

import argparse
import json
import time

import lightgbm as lgb
import pandas as pd

from features import DATA, Encoder, Library, pair_features
from policy import class_names, merge_guard, verdicts
from structure import structure_probs

OUT = DATA / "learned-matcher"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v6")
    ap.add_argument("--structure", default="rel-v6", help="relation-head dir with structure.txt")
    args = ap.parse_args()
    mdir = OUT / "models" / args.model
    t0 = time.time()
    lib = Library(str(DATA / "eval/live-2026-10-03.db"))
    cand = pd.read_csv(DATA / "eval/candidates-rhai-2026-10-03.csv", low_memory=False,
                       usecols=["verdict", "type", "entry_a", "entry_b"])
    cand = cand[cand.verdict.isin(["MERGE", "RELATE", "DISTINCT"])].rename(columns={"verdict": "rhai"}).reset_index(drop=True)
    # Cross-type candidates (20% of the pool) are DISTINCT without scoring,
    # as in the Rhai script; views use the entry's own type, never the pair's.
    same = (cand.entry_a.map(lib.type_of) == cand.type) & (cand.entry_b.map(lib.type_of) == cand.type)
    cross = cand[~same].copy()
    cand = cand[same].reset_index(drop=True)
    views = {}
    for e in set(cand.entry_a) | set(cand.entry_b):
        views[e] = lib.view(sorted(lib.entry_pairs[e]), lib.type_of[e])
    enc = Encoder()
    enc.load(OUT / "enc_cache.npy")
    enc.embed([x for v in views.values() for x in list(v.texts) + [n for _, n in v.names] + list(v.artist_names)])
    enc.save(OUT / "enc_cache.npy")
    print(f"views + vectors {time.time() - t0:.0f}s")
    m = lgb.Booster(model_file=str(mdir / "model.txt"))
    names = m.feature_name()
    th = json.loads((mdir / "thresholds.json").read_text())
    parts = []
    B = 50000
    for k in range(0, len(cand), B):
        chunk = cand.iloc[k:k + B]
        F = pd.DataFrame([pair_features(views[a], views[b], enc.cache) for a, b in zip(chunk.entry_a, chunk.entry_b)])
        X = F[names]
        P = m.predict(X, num_threads=16)
        c = chunk.copy()
        for j, name in enumerate(class_names(P.shape[1])):
            c[f"p_{name}"] = P[:, j]
        c["verdict"] = verdicts(c.type, P, th, guard=merge_guard(F), structure=structure_probs(F, c.type, args.structure))
        parts.append(c)
        print(f"  {k + len(chunk)}/{len(cand)} {time.time() - t0:.0f}s", flush=True)
    cross["verdict"] = "DISTINCT"
    out = pd.concat(parts + [cross], ignore_index=True)
    out.to_parquet(mdir / "pool_preds.parquet")
    print(pd.crosstab([out.type, out.rhai], out.verdict, margins=True).to_string())
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
