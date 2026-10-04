"""Draw a stratified, inclusion-probability-weighted labeling batch from a
softmatch candidate CSV and export evidence cards from the snapshot DB.

Strata = entry type x current Rhai verdict x main_title_sim band. Each sampled
item records its stratum's population size N_h and sample size n_h, so any
metric can be estimated for the whole candidate population with weights
N_h / n_h (Horvitz-Thompson), whatever model is later scored on it.

    uv run --with pandas python train/sample_gold.py \
        --candidates data/eval/candidates-rhai-2026-10-03.csv \
        --db data/eval/live-2026-10-03.db --batch human-v1 --out data/eval/gold
"""

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd

BANDS = [-1.0, 0.5, 0.7, 0.82, 0.9, 1.01]
BAND_NAMES = ["le0.5", "0.5-0.7", "0.7-0.82", "0.82-0.9", "gt0.9"]

# (type, verdict, band or None for all bands) -> n. 100 items total.
ALLOCATION = {
    ("track", "MERGE", None): 18,
    ("track", "RELATE", None): 12,
    ("track", "DISTINCT", "gt0.9"): 14,
    ("track", "DISTINCT", "0.82-0.9"): 6,
    ("track", "DISTINCT", "0.7-0.82"): 5,
    ("track", "DISTINCT", "0.5-0.7"): 5,
    ("track", "DISTINCT", "le0.5"): 4,
    ("artist", "MERGE", None): 8,
    ("artist", "DISTINCT", "gt0.82"): 4,
    ("artist", "DISTINCT", "0.5-0.82"): 4,
    ("artist", "DISTINCT", "le0.5"): 2,
    ("release", "DISTINCT", "gt0.9"): 6,
    ("release", "DISTINCT", "0.7-0.9"): 3,
    ("release", "DISTINCT", "le0.7"): 1,
    ("release_group", "MERGE", None): 4,
    ("release_group", "DISTINCT", "gt0.7"): 3,
    ("release_group", "DISTINCT", "le0.7"): 1,
}

RANGES = {
    "gt0.9": (0.9, 9), "0.82-0.9": (0.82, 0.9), "0.7-0.82": (0.7, 0.82),
    "0.5-0.7": (0.5, 0.7), "le0.5": (-9, 0.5), "gt0.82": (0.82, 9),
    "0.5-0.82": (0.5, 0.82), "0.7-0.9": (0.7, 0.9), "le0.7": (-9, 0.7),
    "gt0.7": (0.7, 9),
}


def stratum_mask(df, typ, verdict, band):
    m = (df["type"] == typ) & (df["verdict"] == verdict)
    if band is not None:
        lo, hi = RANGES[band]
        m &= (df["main_title_sim"] > lo) & (df["main_title_sim"] <= hi)
    return m


def link_for(source, ident):
    if ident.startswith("http"):
        return ident
    return None


def entry_card(con, entry_id):
    cur = con.cursor()
    (etype,) = cur.execute("select entry_type from entry where id=?", (entry_id,)).fetchone()
    srcs = cur.execute(
        "select source, identifier, duration_ms, duration_ms_all, release_date, release_type, primary_type"
        " from entry_source where entry_id=? order by source, identifier", (entry_id,)).fetchall()
    sources, durations, dates, rtypes = [], set(), set(), set()
    for s, i, d, dall, rd, rt, pt in srcs:
        sources.append({"source": s, "id": i, "url": link_for(s, i)})
        if d:
            durations.add(d)
        if dall:
            try:
                durations.update(json.loads(dall))
            except ValueError:
                pass
        if rd:
            dates.add(rd)
        if rt or pt:
            rtypes.add(rt or pt)
    aliases = cur.execute(
        "select distinct a.source, a.name, a.\"primary\" from entry_alias a join entry_source es"
        " on es.source=a.source and es.identifier=a.identifier where es.entry_id=?"
        " order by a.\"primary\" desc, a.source, a.name", (entry_id,)).fetchall()
    card = {
        "entry_id": entry_id,
        "type": etype,
        "aliases": [{"source": s, "name": n, "primary": bool(p)} for s, n, p in aliases][:30],
        "sources": sources[:25],
        "durations": sorted(durations),
        "dates": sorted(dates)[:6],
        "release_types": sorted(rtypes),
    }
    if etype in ("track", "release", "release_group"):
        card["artists"] = name_list(cur, (
            "select distinct es2.entry_id, c.role from contribution c"
            " join entry_source es on es.source=c.source and es.identifier=c.identifier"
            " join entry_source es2 on es2.source=c.artist_source and es2.identifier=c.artist_identifier"
            " where es.entry_id=? and (c.main_artist=1 or c.role in ('vocal','arranger','remixer','Arranged By'))"),
            entry_id, 10)
    if etype == "track":
        card["releases"] = name_list(cur, (
            "select distinct esp.entry_id, coalesce(ec.disc_no,'') || '-' || coalesce(ec.track_no,'') from entry_child ec"
            " join entry_source es on es.source=ec.child_source and es.identifier=ec.child_identifier"
            " join entry_source esp on esp.source=ec.parent_source and esp.identifier=ec.parent_identifier"
            " join entry pe on pe.id=esp.entry_id"
            " where es.entry_id=? and pe.entry_type='release'"), entry_id, 8)
    if etype in ("release", "release_group"):
        card["children"] = name_list(cur, (
            "select distinct esc.entry_id, coalesce(ec.disc_no,'') || '-' || coalesce(ec.track_no,'') from entry_child ec"
            " join entry_source es on es.source=ec.parent_source and es.identifier=ec.parent_identifier"
            " join entry_source esc on esc.source=ec.child_source and esc.identifier=ec.child_identifier"
            " join entry ce on ce.id=esc.entry_id"
            " where es.entry_id=? and ce.entry_type in ('track','release')"
            " order by ec.disc_no, ec.track_no"), entry_id, 30)
    if etype == "artist":
        card["credits"] = name_list(cur, (
            "select distinct es.entry_id, c.role from contribution c"
            " join entry_source ea on ea.source=c.artist_source and ea.identifier=c.artist_identifier"
            " join entry_source es on es.source=c.source and es.identifier=c.identifier"
            " where ea.entry_id=? limit 200"), entry_id, 12)
    return card


def name_list(cur, sql, entry_id, cap):
    out, seen = [], set()
    for other, extra in cur.execute(sql, (entry_id,)).fetchall():
        if other in seen or other == entry_id:
            continue
        seen.add(other)
        out.append({"entry_id": other, "name": best_name(cur, other), "extra": extra})
        if len(out) >= cap:
            break
    return out


def best_name(cur, entry_id):
    row = cur.execute(
        "select a.name from entry_alias a join entry_source es on es.source=a.source and es.identifier=a.identifier"
        " where es.entry_id=? order by a.\"primary\" desc, (a.source in ('youtube','nicovideo','soundcloud','unknown_url')),"
        " a.source, a.name limit 1", (entry_id,)).fetchone()
    return row[0] if row else f"#{entry_id}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261003)
    args = ap.parse_args()

    df = pd.read_csv(args.candidates, low_memory=False)
    df = df[df["verdict"].isin(["MERGE", "RELATE", "DISTINCT"])].reset_index(drop=True)
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    snap = hashlib.sha256(Path(args.db).read_bytes()).hexdigest()

    items = []
    for (typ, verdict, band), n in ALLOCATION.items():
        pop = df[stratum_mask(df, typ, verdict, band)]
        n = min(n, len(pop))
        pick = pop.sample(n=n, random_state=args.seed)
        stratum = f"{typ}/{verdict}/{band or 'all'}"
        for _, r in pick.iterrows():
            ea, eb = int(r["entry_a"]), int(r["entry_b"])
            key = f"{min(ea, eb)}-{max(ea, eb)}"
            items.append({
                "item_id": f"{args.batch}-{key}",
                "batch": args.batch,
                "type": typ,
                "stratum": stratum,
                "stratum_population": int(len(pop)),
                "stratum_sample": int(n),
                "weight": len(pop) / n,
                "rhai": {"verdict": verdict, "kind": None if pd.isna(r["kind"]) else r["kind"],
                         "confidence": float(r["confidence"]), "reason": None if pd.isna(r["reason"]) else r["reason"],
                         "main_title_sim": float(r["main_title_sim"])},
                "snapshot": f"sha256:{snap}",
                "a": entry_card(con, ea),
                "b": entry_card(con, eb),
            })
    # Present in a shuffled order so strata (and the current verdict) aren't
    # guessable from position.
    order = sorted(items, key=lambda it: hashlib.sha1(it["item_id"].encode()).hexdigest())
    for i, it in enumerate(order):
        it["seq"] = i + 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.batch}.items.jsonl").write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in order))
    # Labeling-page payload: no Rhai verdict / stratum (avoid anchoring).
    blind = [{k: it[k] for k in ("item_id", "seq", "type", "a", "b")} for it in order]
    (out / f"{args.batch}.blind.json").write_text(json.dumps(blind, ensure_ascii=False))
    print(f"{len(order)} items -> {out}")
    print(pd.Series([it["stratum"] for it in order]).value_counts().sort_index())


if __name__ == "__main__":
    main()
