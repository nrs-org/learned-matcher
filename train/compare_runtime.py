"""Compare a softmatch run of rhai/match.learned.rhai with the Python pool
predictions (docs/plan-v15-runtime.md, phase 6).

    uv run python compare_runtime.py <softmatch.csv> [--model v17]

Verdicts are compared on candidate pairs present in both (the runtime's
blocking differs from the Rhai dry run the pool came from). SIBLING counts as
DISTINCT: the script writes no edge for siblings.
"""

import argparse

import pandas as pd

from features import DATA

OUT = DATA / "learned-matcher"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--model", default="v17")
    args = ap.parse_args()
    run = pd.read_csv(args.csv, usecols=["verdict", "type", "entry_a", "entry_b"], low_memory=False)
    run = run[run.verdict.isin(["MERGE", "RELATE", "DEFER", "DISTINCT"])]
    pool = pd.read_parquet(OUT / "models" / args.model / "pool_preds.parquet")
    pool["verdict"] = pool.verdict.replace({"SIBLING": "DISTINCT"})
    key = lambda d: d.assign(lo=d[["entry_a", "entry_b"]].min(axis=1), hi=d[["entry_a", "entry_b"]].max(axis=1))
    run, pool = key(run), key(pool)
    pool = pool.drop_duplicates(["lo", "hi"])
    both = run.merge(pool[["lo", "hi", "verdict"]], on=["lo", "hi"], suffixes=("_rust", "_py"))
    print(f"runtime pairs {len(run)}, pool pairs {len(pool)}, shared {len(both)}")
    print("runtime verdicts:", run.verdict.value_counts().to_dict())
    agree = (both.verdict_rust == both.verdict_py).mean()
    print(f"agreement on shared pairs: {agree:.5f}")
    print(pd.crosstab(both.verdict_py, both.verdict_rust, rownames=["python"], colnames=["rust"]).to_string())
    off = both[both.verdict_rust != both.verdict_py]
    if len(off):
        print(off.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
