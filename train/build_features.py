"""Compute features for the generated examples and the human gold items.

    uv run python build_features.py
Writes data/learned-matcher/{train_features,gold_features}.parquet
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from features import DATA, Encoder, Library, pair_features

OUT = DATA / "learned-matcher"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA / "eval/live-2026-10-03.db"))
    ap.add_argument("--examples", default=str(OUT / "examples.jsonl"))
    ap.add_argument("--gold", action="append", default=None)
    args = ap.parse_args()
    gold_batches = args.gold or ["human-v1", "adjudicated-v34", "silver-jev-v1"]

    t0 = time.time()
    lib = Library(args.db)
    print(f"library loaded {time.time() - t0:.1f}s")

    examples = [json.loads(l) for l in Path(args.examples).read_text().splitlines()]
    gold = []
    for b in gold_batches:
        gold += [json.loads(l) for l in (DATA / f"eval/gold/{b}.items.jsonl").read_text().splitlines()]

    # views
    view_cache = {}

    def view(pairs, typ):
        key = (typ, tuple(map(tuple, pairs)))
        if key not in view_cache:
            view_cache[key] = lib.view(pairs, typ)
        return view_cache[key]

    jobs = []
    for e in examples:
        jobs.append(("train", e, view(e["left"], e["type"]), view(e["right"], e["type"])))
    for it in gold:
        pa = lib.entry_pairs[it["a"]["entry_id"]]
        pb = lib.entry_pairs[it["b"]["entry_id"]]
        jobs.append(("gold", it, view(sorted(pa), it["type"]), view(sorted(pb), it["type"])))
    print(f"{len(jobs)} pairs, {len(view_cache)} views built {time.time() - t0:.1f}s")

    enc = Encoder()
    cache_path = OUT / "enc_cache.npy"
    enc.load(cache_path)
    texts = set()
    for v in view_cache.values():
        texts.update(v.texts)
        texts.update(n for _, n in v.names)
        texts.update(v.artist_names)
    t1 = time.time()
    enc.embed(list(texts))
    enc.save(cache_path)
    print(f"embedded {len(texts)} texts {time.time() - t1:.1f}s")

    rows = {"train": [], "gold": []}
    for kind, e, va, vb in jobs:
        f = pair_features(va, vb, enc.cache)
        if kind == "train":
            f.update(tier=e["tier"], type=e["type"], label=e["label"], kind=e["kind"], derived=e["derived"],
                     split=e["split"], entry_a=e["entry_a"], entry_b=e["entry_b"], why=e["meta"].get("why"))
        else:
            f.update(item_id=e["item_id"], type=e["type"], weight=e["weight"], stratum=e["stratum"], batch=e["batch"])
        rows[kind].append(f)
    for kind, r in rows.items():
        df = pd.DataFrame(r)
        df.to_parquet(OUT / f"{kind}_features.parquet")
        print(f"{kind}: {df.shape}")
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
