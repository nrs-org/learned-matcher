"""Label unlabeled candidate-pool pairs with Jev (teacher for distillation).

Strata come from a model's pool predictions vs Rhai, so labels land where the
model is new (merges/relates Rhai didn't make), unsure (DEFER), or near a
boundary. Pairs already in the generated examples and entries in any gold
set are skipped. Tracks: prompt v2.1 (identity + kind + direction);
other types: v1.

    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt uv run python jev_teach.py --model v7
Appends to data/learned-matcher/jev_labels.jsonl (cached; re-runs are free).
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from features import ROOT, Library
from jev_client import Jev, build_view

OUT = ROOT / "data/learned-matcher"

STRATA = [
    ("track new merge", lambda p: (p.type == "track") & (p.verdict == "MERGE") & (p.rhai == "DISTINCT"), 1500),
    ("track new relate", lambda p: (p.type == "track") & (p.verdict == "RELATE") & (p.rhai == "DISTINCT"), 1500),
    ("track rhai merge -> relate", lambda p: (p.type == "track") & (p.verdict == "RELATE") & (p.rhai == "MERGE"), 1000),
    ("track both merge", lambda p: (p.type == "track") & (p.verdict == "MERGE") & (p.rhai == "MERGE"), 500),
    ("track defer", lambda p: (p.type == "track") & (p.verdict == "DEFER"), 2500),
    ("track distinct boundary", lambda p: (p.type == "track") & (p.verdict == "DISTINCT") & (p.p_unrelated < 0.95), 1000),
    ("artist defer", lambda p: (p.type == "artist") & (p.verdict == "DEFER"), 1000),
    ("artist relate", lambda p: (p.type == "artist") & (p.verdict == "RELATE"), 400),
    ("artist merge", lambda p: (p.type == "artist") & (p.verdict == "MERGE"), 100),
    ("release defer", lambda p: (p.type == "release") & (p.verdict == "DEFER"), 900),
    ("release relate", lambda p: (p.type == "release") & (p.verdict == "RELATE"), 500),
    ("release merge", lambda p: (p.type == "release") & (p.verdict == "MERGE"), 50),
    ("rg defer/merge", lambda p: (p.type == "release_group") & p.verdict.isin(["DEFER", "MERGE"]), 50),
]


def _video_only_set():
    import sqlite3
    con = sqlite3.connect(f"file:{ROOT / 'data/eval/live-2026-10-03.db'}?mode=ro", uri=True)
    srcs = {}
    for eid, s in con.execute("select distinct es.entry_id, es.source from entry_source es join entry_alias a on a.source=es.source and a.identifier=es.identifier"):
        srcs.setdefault(eid, set()).add(s)
    return {e for e, ss in srcs.items() if ss <= {"youtube", "nicovideo", "soundcloud"}}


_VID = None


def _one_video(p):
    global _VID
    if _VID is None:
        _VID = _video_only_set()
    return pd.Series([(a in _VID) != (b in _VID) for a, b in zip(p.entry_a, p.entry_b)], index=p.index)


MV_STRATA = [
    ("mv merge", lambda p: (p.type == "track") & (p.verdict == "MERGE") & _one_video(p), 300),
    ("mv relate", lambda p: (p.type == "track") & (p.verdict == "RELATE") & _one_video(p), 500),
    ("mv defer", lambda p: (p.type == "track") & (p.verdict == "DEFER") & _one_video(p), 500),
    ("mv sibling", lambda p: (p.type == "track") & (p.verdict == "SIBLING") & _one_video(p), 200),
]

EVAL_STRATA = [
    # random samples of one model's own verdicts, for an unbiased silver estimate
    ("eval track merge", lambda p: (p.type == "track") & (p.verdict == "MERGE"), 400),
    ("eval track relate", lambda p: (p.type == "track") & (p.verdict == "RELATE"), 250),
    ("eval track defer", lambda p: (p.type == "track") & (p.verdict == "DEFER"), 100),
    ("eval track rhai merge", lambda p: (p.type == "track") & (p.rhai == "MERGE"), 250),
    ("eval artist merge", lambda p: (p.type == "artist") & (p.verdict == "MERGE"), 150),
    ("eval artist rhai merge", lambda p: (p.type == "artist") & (p.rhai == "MERGE"), 100),
    ("eval artist defer", lambda p: (p.type == "artist") & (p.verdict == "DEFER"), 100),
    ("eval release relate", lambda p: (p.type == "release") & (p.verdict == "RELATE"), 80),
    ("eval release defer", lambda p: (p.type == "release") & (p.verdict == "DEFER"), 80),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v7")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every stratum cap")
    ap.add_argument("--max-cost", type=float, default=1.0, help="stop before exceeding this many USD this run")
    ap.add_argument("--strata", choices=["teach", "eval", "mv"], default="teach")
    ap.add_argument("--out", default=str(OUT / "jev_labels.jsonl"))
    ap.add_argument("--exclude-jev", action="store_true", help="also skip pairs already in jev_labels.jsonl")
    args = ap.parse_args()

    pool = pd.read_parquet(OUT / "models" / args.model / "pool_preds.parquet")
    held = set()
    for b in ("human-v1", "adjudicated-v34"):
        for l in open(ROOT / f"data/eval/gold/{b}.items.jsonl"):
            it = json.loads(l)
            held.update((it["a"]["entry_id"], it["b"]["entry_id"]))
    labeled = set()
    for l in open(OUT / "examples.jsonl"):
        e = json.loads(l)
        labeled.add((min(e["entry_a"], e["entry_b"]), max(e["entry_a"], e["entry_b"])))
    if args.exclude_jev and (OUT / "jev_labels.jsonl").exists():
        for l in open(OUT / "jev_labels.jsonl"):
            j = json.loads(l)
            labeled.add((min(j["entry_a"], j["entry_b"]), max(j["entry_a"], j["entry_b"])))
    key = [(min(a, b), max(a, b)) for a, b in zip(pool.entry_a, pool.entry_b)]
    pool = pool[[k not in labeled and k[0] not in held and k[1] not in held for k in key]]
    print(f"eligible pool pairs: {len(pool)}")

    picks = []
    taken = set()
    for name, flt, n in {"teach": STRATA, "eval": EVAL_STRATA, "mv": MV_STRATA}[args.strata]:
        s = pool[flt(pool)]
        if s.empty:
            continue
        s = s[[(a, b) not in taken for a, b in zip(s.entry_a, s.entry_b)]]
        s = s.sample(min(len(s), int(n * args.scale)), random_state=11 if args.strata == "teach" else 23)
        taken.update(zip(s.entry_a, s.entry_b))
        picks.append(s.assign(stratum=name, stratum_population=int(flt(pool).sum())))
    picks = pd.concat(picks).reset_index(drop=True)
    print(picks.groupby("stratum", sort=False).size().to_string())

    lib = Library(str(ROOT / "data/eval/live-2026-10-03.db"))
    jev = Jev(concurrency=16)
    out_path = Path(args.out)
    est_per_call = 1150 * 0.042 / 1e6
    budget_calls = int(args.max_cost / est_per_call)
    if len(picks) > budget_calls:
        print(f"capping to {budget_calls} pairs for max cost ${args.max_cost}")
        picks = picks.iloc[:budget_calls]
    jobs = []
    for r in picks.itertuples():
        va = build_view(lib, sorted(lib.entry_pairs[r.entry_a]), r.type)
        vb = build_view(lib, sorted(lib.entry_pairs[r.entry_b]), r.type)
        jobs.append((va, vb, r.type, "v2.1" if r.type == "track" else "v1"))
    resps = jev.ask_many(jobs)
    n_err = 0
    with out_path.open("a") as f:
        for r, resp in zip(picks.itertuples(), resps):
            if "error" in resp:
                n_err += 1
                continue
            a = resp["answers"]
            rec = {"entry_a": int(r.entry_a), "entry_b": int(r.entry_b), "type": r.type, "stratum": r.stratum,
                   "stratum_population": r.stratum_population, "prompt": "v2.1" if r.type == "track" else "v1",
                   "model": args.model, "model_verdict": r.verdict, "rhai": r.rhai, "p_same": r.p_same,
                   "choice": a["identity"]["choice"], "confidence": a["identity"]["confidence"],
                   "probabilities": a["identity"].get("probabilities"),
                   "kind": a.get("kind", {}).get("choice"), "direction": a.get("direction", {}).get("choice")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"new calls {jev.calls}, cache hits {jev.cache_hits}, errors {n_err}, this run ${jev.cost():.4f}")


if __name__ == "__main__":
    main()
