"""Silver, pool-wide precision estimates for Rhai and a model, using Jev
teacher labels as the judge.

Cells = (type, rhai verdict, model verdict). Each cell's Jev label mix (from
the pairs jev_teach.py sampled in it) is weighted by the cell's population in
the whole candidate pool. Only cells with >= --min-n Jev samples count; the
report says how much of each verdict's population that covers.

Jev is ~92-98% right at the confidences used (jev_audit.py), so these are
estimates with a judge error of a few points, not ground truth.

    uv run python jev_pool_estimate.py --model v7 [--min-conf 0.8]
"""

import argparse
import json

import numpy as np
import pandas as pd

from features import DATA

OUT = DATA / "learned-matcher"
CLS = {"same_identity": "same", "derived": "related", "sibling": "related", "related_variant": "related",
       "unrelated": "unrelated", "different_identity": "not_same", "unsure": "unsure"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v7")
    ap.add_argument("--labels-model", default="v7", help="model whose pool strata the Jev labels were drawn from")
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--dev-only", action="store_true",
                    help="only Jev labels whose pair touches a dev-split entity (for models trained on the jev tier)")
    args = ap.parse_args()

    pool = pd.read_parquet(OUT / "models" / args.model / "pool_preds.parquet")
    jl = pd.DataFrame([json.loads(l) for l in open(OUT / "jev_labels.jsonl")])
    jl = jl[jl.confidence >= args.min_conf]
    jl["jev"] = jl.choice.map(CLS)
    if args.dev_only:
        import hashlib
        is_dev = lambda e: int(hashlib.sha1(f"lm-{e}".encode()).hexdigest(), 16) % 5 == 0
        jl = jl[[is_dev(a) or is_dev(b) for a, b in zip(jl.entry_a, jl.entry_b)]]
        print(f"dev-only Jev labels: {len(jl)}")
    # attach the evaluated model's verdict (may differ from the model that drew the strata)
    jl = jl.merge(pool[["entry_a", "entry_b", "verdict"]].rename(columns={"verdict": "mv"}), on=["entry_a", "entry_b"], how="left")

    pop = pool.groupby(["type", "rhai", "verdict"]).size().rename("pop")
    samp = jl.groupby(["type", "rhai", "mv", "jev"]).size().unstack(fill_value=0)
    samp.index.names = ["type", "rhai", "verdict"]
    cells = samp.join(pop, how="left")
    cells["n"] = samp.sum(1)
    for c in ("same", "related", "unrelated", "not_same", "unsure"):
        if c not in cells:
            cells[c] = 0
    decided = cells[["same", "related", "unrelated", "not_same"]].sum(1).replace(0, np.nan)
    for c in ("same", "related", "unrelated", "not_same"):
        cells[f"f_{c}"] = cells[c] / decided
    print(cells[["pop", "n", "f_same", "f_related", "f_unrelated", "f_not_same", "unsure"]].round(2).to_string())

    def estimate(who, verdict, typ, good):
        if who == "rhai":
            sel = cells[(cells.index.get_level_values("rhai") == verdict)]
            total = pool[(pool.type == typ) & (pool.rhai == verdict)].shape[0]
        else:
            sel = cells[(cells.index.get_level_values("verdict") == verdict)]
            total = pool[(pool.type == typ) & (pool.verdict == verdict)].shape[0]
        sel = sel[(sel.index.get_level_values("type") == typ) & (sel.n >= args.min_n)]
        if sel.empty or total == 0:
            return None
        frac = sum(sel[f"f_{g}"].fillna(0) * sel["pop"] for g in good) / sel["pop"]
        p = float((frac * sel["pop"]).sum() / sel["pop"].sum())
        return p, sel["pop"].sum() / total, int(total)

    print(f"\nsilver precision (Jev judge, min_conf={args.min_conf}); coverage = share of that verdict's pool population in cells with >= {args.min_n} Jev samples")
    for typ in ("track", "artist", "release"):
        for verdict, good in (("MERGE", ["same"]), ("RELATE", ["related"] if typ == "track" else ["related", "not_same"])):
            for who in ("rhai", "model"):
                r = estimate(who, verdict, typ, good)
                if r:
                    print(f"  {typ:8s} {who:5s} {verdict:6s} precision {r[0]:.3f}  (coverage {r[1]:.0%} of {r[2]})")
    d = jl[jl.mv == "DEFER"]
    print(f"\nmodel DEFER pairs judged by Jev: {len(d)}; Jev answers: {d.jev.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
