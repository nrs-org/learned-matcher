"""Build a fixed, model-independent silver test set labeled by Jev.

Same design as the human batch (sample_gold.py): strata = type x Rhai verdict
x title-similarity band, each item weighted N_h / n_h, so evaluate.py gives
population estimates of precision AND recall for any model. Pairs come only
from dev-split entities (the generator's entity hash), which no model trains
on, and the human / adjudicated entities are excluded.

Jev is the judge (prompt v2.1 for tracks, v1 otherwise): treat the result as
silver, a few points of judge error, and never train on it.

    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt uv run python silver_set.py
Writes data/eval/gold/silver-jev-v1.{items,labels}.jsonl
"""

import hashlib
import json

import pandas as pd

from features import DATA, Library
from jev_client import Jev, build_view
from sample_gold import RANGES, entry_card

BATCH = "silver-jev-v1"
SCALE = 15  # human-v1 allocation x SCALE
ALLOC = {
    ("track", "MERGE", None): 18, ("track", "RELATE", None): 12, ("track", "DISTINCT", "gt0.9"): 14,
    ("track", "DISTINCT", "0.82-0.9"): 6, ("track", "DISTINCT", "0.7-0.82"): 5, ("track", "DISTINCT", "0.5-0.7"): 5,
    ("track", "DISTINCT", "le0.5"): 4,
    ("artist", "MERGE", None): 8, ("artist", "DISTINCT", "gt0.82"): 4, ("artist", "DISTINCT", "0.5-0.82"): 4,
    ("artist", "DISTINCT", "le0.5"): 2,
    ("release", "DISTINCT", "gt0.9"): 6, ("release", "DISTINCT", "0.7-0.9"): 3, ("release", "DISTINCT", "le0.7"): 1,
    ("release_group", "MERGE", None): 4, ("release_group", "DISTINCT", "gt0.7"): 3, ("release_group", "DISTINCT", "le0.7"): 1,
}
TRACK_MAP = {"same_identity": "same", "derived": "related", "sibling": "siblings", "unrelated": "different", "unsure": "unsure"}


def is_dev(e):
    return int(hashlib.sha1(f"lm-{e}".encode()).hexdigest(), 16) % 5 == 0


def main():
    db = DATA / "eval/live-2026-10-03.db"
    cand = pd.read_csv(DATA / "eval/candidates-rhai-2026-10-03.csv", low_memory=False)
    cand = cand[cand.verdict.isin(["MERGE", "RELATE", "DISTINCT"])]
    held = set()
    for b in ("human-v1", "adjudicated-v34"):
        for l in open(DATA / f"eval/gold/{b}.items.jsonl"):
            it = json.loads(l)
            held.update((it["a"]["entry_id"], it["b"]["entry_id"]))
    ok = [(is_dev(a) or is_dev(b)) and a not in held and b not in held for a, b in zip(cand.entry_a, cand.entry_b)]
    eligible = cand[ok]
    print(f"eligible dev-entity pairs: {len(eligible)} of {len(cand)}")

    import sqlite3
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    lib = Library(str(db))
    rows = []
    for (typ, verdict, band), n in ALLOC.items():
        m = (eligible["type"] == typ) & (eligible["verdict"] == verdict)
        if band:
            lo, hi = RANGES[band]
            m &= (eligible.main_title_sim > lo) & (eligible.main_title_sim <= hi)
        # population = the whole pool's stratum (eligible pairs are a hash-random subset of it)
        mp = (cand["type"] == typ) & (cand["verdict"] == verdict)
        if band:
            mp &= (cand.main_title_sim > lo) & (cand.main_title_sim <= hi)
        pop = eligible[m]
        k = min(n * SCALE, len(pop))
        for _, r in pop.sample(k, random_state=101).iterrows():
            rows.append((typ, verdict, band, int(mp.sum()), k, r))

    jev = Jev(concurrency=16)
    jobs = []
    for typ, *_, r in rows:
        a, b = int(r.entry_a), int(r.entry_b)
        jobs.append((build_view(lib, sorted(lib.entry_pairs[a]), typ), build_view(lib, sorted(lib.entry_pairs[b]), typ),
                     typ, "v2.1" if typ == "track" else "v1"))
    resps = jev.ask_many(jobs)

    groups = {}
    for (ps, pi, cs, ci) in con.execute("select parent_source, parent_identifier, child_source, child_identifier from entry_child"):
        pe, ce = lib.entry_of.get((ps, pi)), lib.entry_of.get((cs, ci))
        if pe is not None and ce is not None and lib.type_of.get(pe) == "release_group":
            groups.setdefault(ce, set()).add(pe)

    items, labels = [], []
    for (typ, verdict, band, N, n, r), resp in zip(rows, resps):
        if "error" in resp:
            continue
        a, b = int(r.entry_a), int(r.entry_b)
        ans = resp["answers"]["identity"]
        if typ == "track":
            ident = TRACK_MAP[ans["choice"]]
        else:
            ident = {"same_identity": "same", "unsure": "unsure"}.get(ans["choice"], "different")
            if ident == "different" and typ == "release" and groups.get(a, set()) & groups.get(b, set()):
                ident = "related"
        key = f"{min(a, b)}-{max(a, b)}"
        item_id = f"{BATCH}-{key}"
        items.append({"item_id": item_id, "batch": BATCH, "type": typ, "stratum": f"{typ}/{verdict}/{band or 'all'}",
                      "stratum_population": N, "stratum_sample": n, "weight": N / n,
                      "rhai": {"verdict": verdict, "kind": None, "confidence": None, "reason": None,
                               "main_title_sim": float(r.main_title_sim)},
                      "a": {"entry_id": a}, "b": {"entry_id": b}})
        kind = resp["answers"].get("kind", {}).get("choice")
        labels.append({"item_id": item_id, "type": typ, "identity": ident,
                       "kind": None if kind in (None, "not_applicable") else kind,
                       "note": f"jev {ans['choice']} conf={ans['confidence']:.2f}", "confidence": ans["confidence"]})
    out = DATA / "eval/gold"
    (out / f"{BATCH}.items.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items))
    (out / f"{BATCH}.labels.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in labels))
    print(f"{len(items)} items; new calls {jev.calls}, cache hits {jev.cache_hits}, cost ${jev.cost():.4f}")
    print(pd.Series([(i['type'], l['identity']) for i, l in zip(items, labels)]).value_counts().to_string())


if __name__ == "__main__":
    main()
