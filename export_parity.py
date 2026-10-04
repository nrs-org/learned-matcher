"""Export parity fixtures for the Rust runtime of the learned matcher
(docs/plan-v15-runtime.md, phase 1).

    uv run python export_parity.py [--model v15-noorig --rel rel-noorig]

Writes data/learned-matcher/parity/<model>/:
  meta.json        schema, model dirs, feature/class/kind names, thresholds
  facts.jsonl      one per library pair: the `pair_facts_json` contract
  entries.jsonl    {entry_id, type, pairs} (pairs sorted like EntryInfo.pairs)
  pairs.jsonl      one per scored pair: features + model outputs + verdict
  texts.json       encoder inputs, in vectors.f32 row order
  vectors.f32      little-endian f32, len(texts) x 256 (Python ONNX encoder)

The reference features are computed from the facts alone (`view_from_facts`)
and checked against `Library.view` on the same pairs, so the facts carry
everything the model reads. The one deliberate change from training:
`best_name` falls back over an entry's pairs in sorted order, not DB row
order, so the host can reproduce it.
"""

import argparse
import json
import math
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import (DIM, ROOT, VIDEO_SOURCES, Encoder, Library, View, core_title,
                      pair_features)
from policy import class_names, merge_guard, verdicts
from structure import structure_probs

OUT = ROOT / "data/learned-matcher"
SCHEMA = "musiclib-pair-facts/1"


def pair_facts(lib, p):
    """Everything `Library.view` reads about one (source, identifier) pair."""
    ds, rd, _ = lib.src.get(p, (set(), None, None))
    rt = pt = None
    if p in lib.raw_types:
        rt, pt = lib.raw_types[p]
    eid = lib.entry_of.get(p)
    return {
        "source": p[0],
        "identifier": p[1],
        "entry_id": eid,
        "entry_type": lib.type_of.get(eid),
        "names": [[n, prim] for n, prim in lib.names.get(p, [])],
        "durations": sorted(ds),
        "release_date": rd,
        "release_type": rt,
        "primary_type": pt,
        "contributions": [{
            "artist": list(a),
            "artist_entry_id": lib.entry_of.get(a),
            "artist_name": lib.best_name(a),
            "role": role,
            "main": main,
        } for a, role, main in lib.contrib.get(p, [])],
        "parents": [{
            "entry_id": lib.entry_of.get(parent),
            "entry_type": lib.type_of.get(lib.entry_of.get(parent)),
            "disc": dn,
            "track": tn,
        } for parent, dn, tn in lib.parents.get(p, [])],
        "children": [{
            "entry_id": lib.entry_of.get(k),
            "entry_type": lib.type_of.get(lib.entry_of.get(k)),
            "name": lib.names[k][0][0] if lib.names.get(k) else None,
        } for k in lib.kids.get(p, [])],
        "credited": [lib.entry_of.get(item) for item in lib.credited.get(p, [])[:500]],
    }


def primary_artist_name(facts):
    cs = facts["contributions"]
    for c in cs:
        if c["main"] and c["role"] == "listed_artist":
            return c["artist_name"]
    for c in cs:
        if c["role"] == "uploader":
            return c["artist_name"]
    return None


def view_from_facts(pair_facts_list, typ):
    """`Library.view` rebuilt from facts only (MB originality facts excluded:
    the shipped model does not read them)."""
    v = View(type=typ)
    named_sources = set()
    for f in pair_facts_list:
        v.durations |= set(f["durations"])
        rd = f["release_date"]
        if rd and rd[:4].isdigit():
            v.years.add(int(rd[:4]))
        rt = f["release_type"] or f["primary_type"]
        if rt:
            v.rtypes.add(rt.lower())
        pa_name = primary_artist_name(f) if typ == "track" else None
        for n, _ in f["names"]:
            v.names.append((f["source"], n))
            v.texts.append(f"{n} [A] {pa_name}" if pa_name else n)
            named_sources.add(f["source"])
        for c in f["contributions"]:
            if c["main"] or c["role"] in ("listed_artist", "uploader", "vocal"):
                if c["artist_entry_id"] is not None:
                    v.artists.add(c["artist_entry_id"])
                if c["artist_name"]:
                    v.artist_names.add(c["artist_name"])
        for c in f["contributions"]:
            if c["role"] == "uploader" and c["artist_entry_id"] is not None:
                v.uploaders.add(c["artist_entry_id"])
        for par in f["parents"]:
            pe = par["entry_id"]
            if pe is not None and typ == "release" and par["entry_type"] == "release_group":
                v.groups.add(pe)
            if pe is not None and par["entry_type"] == "release":
                v.releases.add(pe)
                if par["track"] is not None:
                    v.positions.add((pe, par["disc"] or 1, par["track"]))
        for k in f["children"]:
            if k["entry_id"] is not None and k["entry_type"] in ("track", "release"):
                v.children.add(k["entry_id"])
            if typ == "release" and k["name"] is not None:
                v.child_titles.append(core_title(k["name"]))
        if typ == "artist":
            v.credits |= {e for e in f["credited"] if e is not None}
    seen, names, texts = set(), [], []
    for (s, n), t in zip(v.names, v.texts):
        if t in seen:
            continue
        seen.add(t)
        names.append((s, n))
        texts.append(t)
    v.names, v.texts = names[:40], texts[:40]
    v.has_video = bool(named_sources & VIDEO_SOURCES)
    v.video_only = bool(named_sources) and named_sources <= VIDEO_SOURCES
    return v


def load_library(db):
    lib = Library(db)
    # Deterministic best_name fallback (sorted pairs, like EntryInfo.pairs).
    lib.entry_pairs = {e: sorted(ps) for e, ps in lib.entry_pairs.items()}
    # Library folds release_type/primary_type into one value; keep both raw.
    import sqlite3
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    lib.raw_types = {(s, i): (rt, pt) for s, i, rt, pt in
                     con.execute("select source, identifier, release_type, primary_type from entry_source")}
    return lib


def sample_pairs(model, per_stratum, seed):
    gold = []
    for b in ["human-v1", "adjudicated-v34", "silver-jev-v1"]:
        for line in (ROOT / f"data/eval/gold/{b}.items.jsonl").read_text().splitlines():
            it = json.loads(line)
            gold.append((it["a"]["entry_id"], it["b"]["entry_id"], it["type"], f"gold:{b}"))
    pool = pd.read_parquet(OUT / "models" / model / "pool_preds.parquet")
    picks = pool.sample(frac=1, random_state=seed).groupby(["type", "verdict"]).head(per_stratum)
    out, seen = [], set()
    for a, b, t, origin in gold + [(r.entry_a, r.entry_b, r.type, f"pool:{r.verdict}")
                                   for r in picks.itertuples()]:
        key = (int(a), int(b))
        if key not in seen:
            seen.add(key)
            out.append((int(a), int(b), t, origin))
    return out


def nan_to_none(x):
    return None if isinstance(x, float) and math.isnan(x) else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v15-noorig")
    ap.add_argument("--rel", default="rel-noorig")
    ap.add_argument("--db", default=str(ROOT / "data/eval/live-2026-10-03.db"))
    ap.add_argument("--per-stratum", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = OUT / "parity" / args.model
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    lib = load_library(args.db)
    jobs = sample_pairs(args.model, args.per_stratum, args.seed)
    # Cross-type pairs never reach the model (the script returns DISTINCT).
    n_all = len(jobs)
    jobs = [j for j in jobs if lib.type_of.get(j[0]) == j[2] == lib.type_of.get(j[1])]
    print(f"dropped {n_all - len(jobs)} cross-type pairs")
    entries = {}
    for a, b, t, _ in jobs:
        entries[a] = entries[b] = t
    entry_pairs = {e: sorted(lib.entry_pairs[e]) for e in entries}
    facts = {p: pair_facts(lib, p) for ps in entry_pairs.values() for p in ps}
    print(f"{len(jobs)} pairs, {len(entries)} entries, {len(facts)} library pairs ({time.time() - t0:.0f}s)")

    views = {e: view_from_facts([facts[p] for p in entry_pairs[e]], t) for e, t in entries.items()}
    enc = Encoder()
    enc.load(OUT / "enc_cache.npy")
    texts = sorted({x for v in views.values()
                    for x in list(v.texts) + [n for _, n in v.names] + list(v.artist_names)})
    enc.embed(texts)

    F = pd.DataFrame([pair_features(views[a], views[b], enc.cache) for a, b, _, _ in jobs])

    # The facts must carry everything: Library.view gives the same features.
    lib_views = {e: lib.view(entry_pairs[e], t) for e, t in entries.items()}
    G = pd.DataFrame([pair_features(lib_views[a], lib_views[b], enc.cache) for a, b, _, _ in jobs])
    cols = [c for c in F.columns if "orig" not in c]
    diff = ~((F[cols] == G[cols]) | (F[cols].isna() & G[cols].isna()))
    if diff.to_numpy().any():
        bad = diff.any()
        raise SystemExit(f"facts views disagree with Library.view on: {list(bad[bad].index)}")
    print(f"facts-only features == Library.view features ({len(cols)} columns)")

    # Main model, policy, relation heads.
    mdir, rdir = OUT / "models" / args.model, OUT / "models" / args.rel
    m = lgb.Booster(model_file=str(mdir / "model.txt"))
    th = json.loads((mdir / "thresholds.json").read_text())
    types = pd.Series([t for _, _, t, _ in jobs])
    P = m.predict(F[m.feature_name()])
    guard = merge_guard(F)
    p_sib = structure_probs(F, types, args.rel)
    verdict = verdicts(types, P, th, guard=guard, structure=p_sib)

    km = lgb.Booster(model_file=str(rdir / "kind.txt"))
    kinds = json.loads((rdir / "kinds.json").read_text())
    KX = pd.DataFrame({c: (F[c[4:]].abs() if c.startswith("abs_") else F[c]) for c in km.feature_name()})
    PK = km.predict(KX)
    dm = lgb.Booster(model_file=str(rdir / "direction.txt"))
    DX = F[dm.feature_name()]
    DXf = DX.copy()
    signed = [c for c in dm.feature_name() if c.startswith("s_")]
    DXf[signed] = -DXf[signed]
    p_a_derived = (dm.predict(DX) + 1 - dm.predict(DXf)) / 2

    feature_names = list(F.columns)
    meta = {
        "schema": SCHEMA,
        "model": args.model,
        "rel": args.rel,
        "db": str(args.db),
        "feature_names": feature_names,
        "model_features": m.feature_name(),
        "classes": class_names(P.shape[1]),
        "kinds": kinds,
        "thresholds": th,
        "dim": DIM,
        "texts": len(texts),
        "notes": "features are NaN-as-null; orig_* features are null by design (not shipped)",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    with open(out / "facts.jsonl", "w") as fh:
        for p in sorted(facts):
            fh.write(json.dumps(facts[p], ensure_ascii=False) + "\n")
    with open(out / "entries.jsonl", "w") as fh:
        for e in sorted(entries):
            fh.write(json.dumps({"entry_id": e, "type": entries[e],
                                 "pairs": [list(p) for p in entry_pairs[e]]}, ensure_ascii=False) + "\n")
    with open(out / "pairs.jsonl", "w") as fh:
        for k, (a, b, t, origin) in enumerate(jobs):
            row = {
                "entry_a": a, "entry_b": b, "type": t, "origin": origin,
                "features": [nan_to_none(float(F.iat[k, j])) for j in range(len(feature_names))],
                "p": [float(x) for x in P[k]],
                "guard": bool(guard[k]),
                "p_sibling_structure": p_sib[k],
                "kind_p": [float(x) for x in PK[k]],
                "p_a_derived": float(p_a_derived[k]),
                "verdict": verdict[k],
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "texts.json").write_text(json.dumps(texts, ensure_ascii=False))
    np.stack([enc.cache[t] for t in texts]).astype("<f4").tofile(out / "vectors.f32")

    print(pd.crosstab(pd.Series([o.split(":")[0] for _, _, _, o in jobs], name="origin"),
                      pd.Series(verdict, name="verdict")).to_string())
    print(f"wrote {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
