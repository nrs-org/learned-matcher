"""Audit Jev (prompt v1 = shipped, v2 = ontology-aligned) against the
cleanest generated labels and the LLM-adjudicated set, next to the learned
model, per category. Answers are cached; re-running is free.

    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt uv run python jev_audit.py --model v6
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import ROOT, Library
from jev_client import Jev, build_view

OUT = ROOT / "data/learned-matcher"
V1_MAP = {"same_identity": "same", "related_variant": "related", "unrelated": "unrelated",
          "different_identity": "unrelated", "unsure": "unsure"}
V2_MAP = {"same_identity": "same", "derived": "related", "sibling": "sibling", "unrelated": "unrelated",
          "different_identity": "unrelated", "unsure": "unsure"}

# name: (type, filter, n)
CATEGORIES = {
    "T1 MV = same (MB music video, full)": ("track", lambda d: (d.why == "mb_rec_rec") & (d.label == "same"), 30),
    "T2 MB rec relation (instr/remix/edit)": ("track", lambda d: (d.why == "mb_rec_rec") & (d.label == "related"), 30),
    "T3 cover vs original": ("track", lambda d: (d.why == "mb_work_attr") & (d.kind == "cover"), 25),
    "T4 live vs studio": ("track", lambda d: (d.why == "mb_work_attr") & (d.kind == "live"), 20),
    "T5 sibling versions": ("track", lambda d: d.why == "mb_work_siblings", 25),
    "T6 hard unrelated (MB different works, similar titles)": ("track", lambda d: (d.why == "mb_work_disjoint") & (d.enc_title_max >= 0.75), 40),
    "T7 split-half same": ("track", lambda d: (d.tier == "probe"), 30),
    "T8 same-uploader different song": ("track", lambda d: d.tier == "uploader_neg", 20),
    "T9 model DEFER band (any label)": ("track", lambda d: d.model == "DEFER", 40),
    "T10 model RELATE (any label)": ("track", lambda d: d.model == "RELATE", 40),
    "A1 artist split-half same": ("artist", lambda d: d.tier == "probe", 25),
    "A2 artist MB relation (member/persona)": ("artist", lambda d: d.why == "mb_artist_rel", 15),
    "A3 artist MB distinct, similar names": ("artist", lambda d: (d.why == "mb_artist_distinct") & (d.enc_max >= 0.8), 30),
    "R1 release split-half same": ("release", lambda d: d.tier == "probe", 15),
    "R2 release same group (editions)": ("release", lambda d: d.why == "mb_same_rg", 20),
    "R3 release different groups, similar titles": ("release", lambda d: (d.why == "mb_rg_disjoint") & (d.enc_max >= 0.8), 20),
}


from policy import class_names, merge_guard, verdicts
from structure import structure_probs

VMAP = {"MERGE": "same", "RELATE": "related", "SIBLING": "sibling", "DISTINCT": "unrelated", "DEFER": "defer"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v6")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    mdir = OUT / "models" / args.model
    th = json.loads((mdir / "thresholds.json").read_text())

    lib = Library(str(ROOT / "data/eval/live-2026-10-03.db"))
    examples = [json.loads(l) for l in (OUT / "examples.jsonl").read_text().splitlines()]
    feats = pd.read_parquet(OUT / "train_features.parquet", columns=["enc_title_max", "enc_max"])
    dev = pd.read_parquet(mdir / "dev_preds.parquet")
    allf = pd.read_parquet(OUT / "train_features.parquet")
    dev = dev.join(feats)
    dev["kind"] = [examples[i]["kind"] for i in dev.index]
    pcols = [c for c in ("p_same", "p_related", "p_sibling", "p_unrelated") if c in dev]
    F = allf.loc[dev.index]
    dev["model"] = verdicts(dev.type, dev[pcols].to_numpy(), th, guard=merge_guard(F), structure=structure_probs(F, dev.type))

    rows = []
    for cat, (typ, flt, n) in CATEGORIES.items():
        pool = dev[(dev.type == typ) & flt(dev)]
        for i, r in pool.sample(min(n, len(pool)), random_state=args.seed).iterrows():
            e = examples[i]
            rows.append({"set": "generated", "category": cat, "type": typ, "gold": e["label"], "kind": e["kind"],
                         "left": e["left"], "right": e["right"], "model": VMAP[r.model], "p_same": r.p_same})
    # adjudicated (LLM labels) with model predictions on full entries
    gp = pd.read_csv(mdir / "gold_preds.csv").set_index("item_id")
    gp = gp[gp.index.str.startswith("adjudicated")]
    labs = {json.loads(l)["item_id"]: json.loads(l)["identity"] for l in open(ROOT / "data/eval/gold/adjudicated-v34.labels.jsonl")}
    for l in open(ROOT / "data/eval/gold/adjudicated-v34.items.jsonl"):
        it = json.loads(l)
        g = {"same": "same", "related": "related", "different": "unrelated"}[labs[it["item_id"]]]
        p = gp.loc[it["item_id"]]
        rows.append({"set": "adjudicated", "category": f"ADJ {it['type']}", "type": it["type"], "gold": g, "kind": None,
                     "left": sorted(lib.entry_pairs[it["a"]["entry_id"]]), "right": sorted(lib.entry_pairs[it["b"]["entry_id"]]),
                     "model": VMAP[p.verdict], "p_same": p.p_same})
    df = pd.DataFrame(rows)

    jev = Jev()
    jobs, keys = [], []
    for i, r in df.iterrows():
        va, vb = build_view(lib, r.left, r.type), build_view(lib, r.right, r.type)
        for ver in (("v1", "v2") if r.type == "track" else ("v1",)):
            jobs.append((va, vb, r.type, ver))
            keys.append((i, ver))
    print(f"{len(df)} pairs, {len(jobs)} Jev requests")
    resps = jev.ask_many(jobs)
    errors = 0
    for (i, ver), resp in zip(keys, resps):
        if "error" in resp:
            errors += 1
            continue
        a = resp["answers"]["identity"]
        mp = V1_MAP if ver == "v1" else V2_MAP
        df.loc[i, f"jev_{ver}"] = mp[a["choice"]]
        df.loc[i, f"jev_{ver}_raw"] = a["choice"]
        df.loc[i, f"jev_{ver}_conf"] = a["confidence"]
        if ver == "v2" and "kind" in resp["answers"]:
            df.loc[i, "jev_v2_kind"] = resp["answers"]["kind"]["choice"]
            df.loc[i, "jev_v2_dir"] = resp["answers"]["direction"]["choice"]
    df.loc[df.type != "track", "jev_v2"] = df.loc[df.type != "track", "jev_v1"]
    df.loc[df.type != "track", "jev_v2_conf"] = df.loc[df.type != "track", "jev_v1_conf"]
    print(f"new calls {jev.calls}, cache hits {jev.cache_hits}, errors {errors}, cost ${jev.cost():.4f}")

    def acc(pred, gold, nontrack):
        # non-track Jev has no "related" answer: score same vs not-same there
        if nontrack:
            return ((pred == "same") == (gold == "same"))
        return pred == gold

    lines = []
    hdr = f"{'category':55s} {'n':>3s} | {'jev v1':>7s} {'unsure':>6s} | {'jev v2':>7s} {'unsure':>6s} | {'model':>6s} {'defer':>6s}"
    lines.append(hdr)
    for cat, g in df.groupby("category", sort=False):
        nt = g.type.iloc[0] != "track"
        out = [f"{cat:55s} {len(g):3d}"]
        for col in ("jev_v1", "jev_v2", "model"):
            unsure_val = "defer" if col == "model" else "unsure"
            decided = g[g[col] != unsure_val]
            a = acc(decided[col], decided.gold, nt).mean() if len(decided) else np.nan
            out.append(f"{a:7.2f} {1 - len(decided) / len(g):6.0%}")
        lines.append(" | ".join(out))
    report = "\n".join(lines)
    print(report)
    df.drop(columns=["left", "right"]).to_csv(OUT / f"jev_audit_{args.model}.csv", index=False)
    (OUT / f"jev_audit_{args.model}.txt").write_text(report)


if __name__ == "__main__":
    main()
