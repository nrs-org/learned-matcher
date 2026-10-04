"""Resolve the September LLM-adjudicated labels (pilot-v3 + active-v4) onto
the live snapshot and write them in the gold format as batch `adjudicated-v34`.

These are LLM labels with blind adjudication, not human labels: an interim
yardstick until the human batch exists, reported as its own tier. They were
not sampled from the current candidate pool, so every item has weight 1.
"""

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SNAP = DATA / "eval/live-2026-10-03.db"
LABEL = {"same_identity": "same", "different_identity": "different"}


def main():
    con = sqlite3.connect(f"file:{SNAP}?mode=ro", uri=True)
    entry_of = {(s, i): e for s, i, e in con.execute("select source, identifier, entry_id from entry_source")}
    types = dict(con.execute("select id, entry_type from entry"))
    import pandas as pd
    cand = pd.read_csv(DATA / "eval/candidates-rhai-2026-10-03.csv", low_memory=False, usecols=["verdict", "entry_a", "entry_b"])
    rhai = {(min(a, b), max(a, b)): v for v, a, b in zip(cand.verdict, cand.entry_a, cand.entry_b)}

    judg = {}
    for f in ("dedup-entry-pilot-v3.effective.jsonl", "dedup-entry-active-v4.effective.jsonl"):
        for l in (DATA / f).read_text().splitlines():
            d = json.loads(l)
            judg[d["item_id"]] = d["judgment"]
    items, labels, stats = [], [], {"total": 0, "unresolved": 0, "same_entry": 0, "insufficient": 0}
    seen = set()
    for f in ("dedup-entry-pilot-v3.private.jsonl", "dedup-entry-active-v4.private.jsonl"):
        for l in (DATA / f).read_text().splitlines():
            p = json.loads(l)
            stats["total"] += 1
            j = judg.get(p["item_id"])
            if not j:
                continue
            rel = j["factual"]["entity_relation"]
            sides = []
            for side in ("left", "right"):
                ents = set()
                for r in p[side]["records"]:
                    src, _, ident = r["record_id"].partition(":")
                    e = entry_of.get((src, ident))
                    if e is not None:
                        ents.add(e)
                sides.append(ents)
            if len(sides[0]) != 1 or len(sides[1]) != 1:
                stats["unresolved"] += 1
                continue
            a, b = next(iter(sides[0])), next(iter(sides[1]))
            if a == b:
                stats["same_entry"] += 1  # already unified by hard-ID linking
                continue
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            if rel not in LABEL:
                stats["insufficient"] += 1
                continue
            ident = LABEL[rel]
            if ident == "different" and j["factual"].get("relations"):
                ident = "related"
            typ = types[a]
            item_id = f"adjudicated-v34-{key[0]}-{key[1]}"
            v = rhai.get(key)
            items.append({
                "item_id": item_id, "batch": "adjudicated-v34", "type": typ, "stratum": f"{typ}/adjudicated",
                "stratum_population": 1, "stratum_sample": 1, "weight": 1.0,
                "rhai": {"verdict": v if v in ("MERGE", "RELATE", "DISTINCT") else "DISTINCT",
                         "retrieved": v is not None, "kind": None, "confidence": None, "reason": None, "main_title_sim": None},
                "a": {"entry_id": a}, "b": {"entry_id": b},
            })
            labels.append({"item_id": item_id, "type": typ, "identity": ident, "kind": None,
                           "note": "LLM-adjudicated (pilot-v3/active-v4)"})
    out = DATA / "eval/gold"
    (out / "adjudicated-v34.items.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items))
    (out / "adjudicated-v34.labels.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in labels))
    print(stats, "resolved:", len(items))
    from collections import Counter
    print(Counter((x["type"], y["identity"]) for x, y in zip(items, labels)))
    print("retrieved by current blocking:", sum(x["rhai"]["retrieved"] for x in items))


if __name__ == "__main__":
    main()
