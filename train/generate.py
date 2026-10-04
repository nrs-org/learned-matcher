"""Generate labeled training/dev pairs from the library snapshot + MB facts.

Every example is a pair of *views*; a view is a set of (source, identifier)
pairs, so the same feature code handles whole entries and split halves.

Tiers (kept separate in every report):
  probe     same_identity: one multi-provider entry cut into two disjoint
            provider halves (positives in the library's own distribution).
  probe_neg unrelated: halves of two entries the MB tier labels unrelated,
            so the "half-sized view" shape is balanced across classes.
  mb        candidate-pool pairs labeled from MusicBrainz facts:
            recording<->recording relations (music video / edit / remix /
            karaoke / instrumental / a cappella), shared works with
            performance attributes (live / cover / ...), disjoint works
            (unrelated), artist<->artist relations, release groups.
  provider  library entry_relation rows imported from MB (cover / remix /
            arrangement).

Unlabeled candidate pairs are never used as negatives: unlinked duplicates
exist among them (positive-unlabeled), concentrated where it matters most.

Output: data/learned-matcher/examples.jsonl
"""

import argparse
import hashlib
import json
import random
import sqlite3
from collections import defaultdict
from pathlib import Path

import pandas as pd

import features as F

from paths import DATA
MB = DATA / "learned-matcher/mb"
NAMED_SOURCES_SKIP = {"isrc", "upc", "unknown_url"}  # never carry names on their own

# MB recording<->recording: name -> (label, kind, which side is derived: 0/1/None)
REC_REC = {
    "music video": ("mv", None, 1),            # entity1 is the MV of entity0
    "edit": ("related", "edit", 0),            # entity0 is an edit of entity1
    "remix": ("related", "remix", 0),          # entity0 is a remix of entity1
    "karaoke": ("related", "instrumental", 1),  # entity0 has karaoke version entity1
    "instrumental": ("related", "instrumental", 1),
    "a cappella": ("related", "other", 1),
}
PERF_KINDS = {"live": "live", "cover": "cover", "instrumental": "instrumental", "karaoke": "instrumental"}
SKIP_ATTRS = {"medley", "partial"}
ARTIST_RELATED = {"member of band", "subgroup", "is person", "voice actor", "perform as"}


def load_entries(con):
    pairs = defaultdict(list)
    types = {}
    for eid, t in con.execute("select id, entry_type from entry"):
        types[eid] = t
    for s, i, eid in con.execute("select source, identifier, entry_id from entry_source"):
        pairs[eid].append((s, i))
    named = set(con.execute("select distinct source || char(0) || identifier from entry_alias").fetchall())
    named = {x[0] for x in named}
    durations = {}
    for s, i, d, dall in con.execute("select source, identifier, duration_ms, duration_ms_all from entry_source"):
        ds = set()
        if d:
            ds.add(d)
        if dall:
            try:
                ds.update(json.loads(dall))
            except ValueError:
                pass
        durations[(s, i)] = ds
    return types, pairs, named, durations


def mbid_of(identifier):
    # https://musicbrainz.org/<kind>/<uuid>
    parts = identifier.rstrip("/").split("/")
    return parts[-2], parts[-1]


def split_halves(entry_pairs, named, rng):
    """Partition an entry's pairs by provider into two halves that each carry
    at least one name. Returns None when the entry has < 2 named providers."""
    by_src = defaultdict(list)
    for p in entry_pairs:
        by_src[p[0]].append(p)
    named_srcs = sorted(s for s, ps in by_src.items()
                        if s not in NAMED_SOURCES_SKIP and any(f"{a}\x00{b}" in named for a, b in ps))
    if len(named_srcs) < 2:
        return None
    rng.shuffle(named_srcs)
    cut = rng.randint(1, len(named_srcs) - 1)
    left_srcs, right_srcs = set(named_srcs[:cut]), set(named_srcs[cut:])
    left, right = [], []
    for s, ps in by_src.items():
        if s in left_srcs:
            left += ps
        elif s in right_srcs:
            right += ps
        else:  # nameless ids (isrc, upc, crawled urls): attach to a random side
            (left if rng.random() < 0.5 else right).extend(ps)
    return sorted(left), sorted(right)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA / "eval/live-2026-10-03.db"))
    ap.add_argument("--candidates", default=str(DATA / "eval/candidates-rhai-2026-10-03.csv"))
    ap.add_argument("--out", default=str(DATA / "learned-matcher/examples.jsonl"))
    ap.add_argument("--gold-items", action="append",
                    default=[str(DATA / f"eval/gold/{b}.items.jsonl") for b in ("human-v1", "adjudicated-v34")])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--jev", default=str(DATA / "learned-matcher/jev_labels.jsonl"),
                    help="Jev teacher labels to add as tier `jev` ('' to skip)")
    ap.add_argument("--jev-min-conf-track", type=float, default=0.9)
    ap.add_argument("--jev-min-conf-other", type=float, default=0.8)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    types, pairs, named, durations = load_entries(con)

    # Entries touched by the human test set never enter training.
    held_out = set()
    for path in args.gold_items:
        for line in Path(path).read_text().splitlines():
            it = json.loads(line)
            held_out.update((it["a"]["entry_id"], it["b"]["entry_id"]))

    # MB gid -> entry ids, per kind
    mb_entries = defaultdict(set)
    entry_mb = defaultdict(lambda: defaultdict(set))
    for eid, ps in pairs.items():
        for s, i in ps:
            if s == "musicbrainz":
                kind, gid = mbid_of(i)
                mb_entries[(kind, gid)].add(eid)
                entry_mb[eid][kind].add(gid)

    rec = pd.read_csv(MB / "rec.tsv", sep="\t", header=None, names=["gid", "length", "video", "name", "ac"], quoting=3, dtype=str)
    rec_video = dict(zip(rec.gid, rec.video == "t"))
    rec_name = dict(zip(rec.gid, rec.name.fillna("")))
    rec_len = {g: (int(l) if isinstance(l, str) and l.isdigit() else None) for g, l in zip(rec.gid, rec.length)}
    rec_works = defaultdict(dict)  # rec gid -> {work gid: attrs}
    for line in (MB / "rec_work.tsv").read_text().splitlines():
        g, w, attrs = (line.split("\t") + [""])[:3]
        a = set(filter(None, attrs.split(",")))
        rec_works[g][w] = rec_works[g].get(w, set()) | a
    rec_rel = {}
    for line in (MB / "rec_rec.tsv").read_text().splitlines():
        g0, g1, name = line.split("\t")
        rec_rel[(g0, g1)] = name
    art_rel = {}
    for line in (MB / "artist_artist.tsv").read_text().splitlines():
        g0, g1, name = line.split("\t")
        art_rel[(g0, g1)] = name
    release_rg = dict(l.split("\t") for l in (MB / "release_rg.tsv").read_text().splitlines())
    rec_artists = {g: set(a.split(",")) for g, a in (l.split("\t") for l in (MB / "rec_artists.tsv").read_text().splitlines())}
    rec_year = {g: int(y) for g, y in (l.split("\t") for l in (MB / "rec_year.tsv").read_text().splitlines())}
    work_year = {g: int(y) for g, y in (l.split("\t") for l in (MB / "work_year.tsv").read_text().splitlines())}

    examples = []

    def add(tier, typ, left, right, label, kind=None, derived=None, ea=None, eb=None, meta=None):
        examples.append({
            "tier": tier, "type": typ, "label": label, "kind": kind,
            "derived": derived,  # "left" / "right" / None
            "left": left, "right": right, "entry_a": ea, "entry_b": eb, "meta": meta or {},
        })

    # ── probe positives ─────────────────────────────────────────────────────
    probe_halves = {}
    for eid, ps in pairs.items():
        if eid in held_out:
            continue
        halves = split_halves(ps, named, rng)
        if halves is None:
            continue
        probe_halves[eid] = halves
        add("probe", types[eid], halves[0], halves[1], "same", ea=eid, eb=eid)

    # ── MB labels on the real candidate pool ────────────────────────────────
    cand = pd.read_csv(args.candidates, low_memory=False, usecols=["verdict", "type", "entry_a", "entry_b"])
    cand = cand[cand.verdict.isin(["MERGE", "RELATE", "DISTINCT"])]
    seen = set()

    def track_label(a, b):
        ra, rb = entry_mb[a]["recording"], entry_mb[b]["recording"]
        if not ra or not rb:
            return None
        labels = set()
        for x in ra:
            for y in rb:
                for (g0, g1, flip) in ((x, y, False), (y, x, True)):
                    name = rec_rel.get((g0, g1))
                    if name not in REC_REC:
                        continue
                    lab, kind, derived_idx = REC_REC[name]
                    # derived side in terms of (a, b)
                    derived = None
                    if derived_idx is not None:
                        derived_is_g0 = derived_idx == 0
                        derived = ("a" if derived_is_g0 else "b") if not flip else ("b" if derived_is_g0 else "a")
                    if lab == "mv":
                        # An MV is the same identity only when it can stand in
                        # for the audio: the full song. Short / dance-shot /
                        # other "ver." MVs are related (user rule, 2026-10-03).
                        audio, mv = (g0, g1)
                        la, lm = rec_len.get(audio), rec_len.get(mv)
                        mv_markers, mv_named = F.markers([rec_name.get(mv, "")])
                        au_markers, au_named = F.markers([rec_name.get(audio, "")])
                        extra = (mv_markers - au_markers) - {"mv", "full"}
                        if "short" in extra or (la and lm and lm < la - 15000):
                            labels.add(("related", "edit", derived))  # short MV / cut
                        elif extra or (mv_named - au_named):
                            labels.add(("related", "alt_version", derived))  # dance shot ver. etc.
                        elif la and lm and lm > la + 90000:
                            labels.add(("unsure", None, None))  # drama MV etc.
                        else:
                            labels.add(("same", None, None))
                    else:
                        labels.add((lab, kind, derived))
        if not labels:
            wa = {w: at for g in ra for w, at in rec_works.get(g, {}).items()}
            wb = {w: at for g in rb for w, at in rec_works.get(g, {}).items()}
            if not wa or not wb:
                return None
            shared = set(wa) & set(wb)
            if not shared:
                if any(at & SKIP_ATTRS for at in list(wa.values()) + list(wb.values())):
                    return None
                return ("unrelated", None, None), "mb_work_disjoint"
            if len(shared) > 1:
                return None
            w = next(iter(shared))
            aa, ab = wa[w] - {"demo"}, wb[w] - {"demo"}
            if (aa | ab) & SKIP_ATTRS:
                return None
            ka = {PERF_KINDS[x] for x in aa if x in PERF_KINDS}
            kb = {PERF_KINDS[x] for x in ab if x in PERF_KINDS}
            # The original is the work's earliest recording (user's sibling
            # rule: versions link to a common original). When both sides'
            # first-release years are known, decide structure from the gap to
            # the work's first year rather than from MB's sparse attributes.
            ya = [rec_year[x] for x in ra if x in rec_year]
            yb = [rec_year[x] for x in rb if x in rec_year]
            # performers compared by MB artist ids (credit names differ by script)
            ac_a = set().union(*[rec_artists.get(x, set()) for x in ra])
            ac_b = set().union(*[rec_artists.get(x, set()) for x in rb])
            other_performer = bool(ac_a) and bool(ac_b) and not (ac_a & ac_b)
            if ya and yb and w in work_year and (other_performer or ka or kb):
                ga, gb = min(ya) - work_year[w], min(yb) - work_year[w]
                kind = sorted(ka | kb)[0] if (ka | kb) else None
                if min(ga, gb) >= 2:
                    return ("sibling", kind, None), "mb_work_gap_siblings"
                if min(ga, gb) <= 1 and max(ga, gb) >= 2:
                    return ("related", kind, "a" if ga > gb else "b"), "mb_work_gap_derived"
            if ka == kb:
                if not ka:
                    return None  # same work, no attributes: dup / re-take / MV — ambiguous
                return ("sibling", sorted(ka)[0], None), "mb_work_siblings"
            if ka and not kb:
                return ("related", sorted(ka)[0], "a"), "mb_work_attr"
            if kb and not ka:
                return ("related", sorted(kb)[0], "b"), "mb_work_attr"
            return ("sibling", sorted(ka | kb)[0], None), "mb_work_siblings"
        if len(labels) > 1:
            labels.discard(("unsure", None, None))
        if len(labels) != 1:
            return None
        return next(iter(labels)), "mb_rec_rec"

    def other_label(typ, a, b):
        kind = {"artist": "artist", "release": "release", "release_group": "release-group"}[typ]
        ga, gb = entry_mb[a][kind], entry_mb[b][kind]
        if not ga or not gb:
            return None
        if typ == "artist":
            rel = [art_rel.get((x, y)) or art_rel.get((y, x)) for x in ga for y in gb]
            rel = [r for r in rel if r]
            if any(r in ARTIST_RELATED for r in rel):
                return ("related", None, None), "mb_artist_rel"
            if rel:
                return None  # sibling / married / collaboration...: different but noisy
            return ("unrelated", None, None), "mb_artist_distinct"
        if typ == "release":
            rga = {release_rg.get(x) for x in ga} - {None}
            rgb = {release_rg.get(x) for x in gb} - {None}
            if not rga or not rgb:
                return None
            if rga & rgb:
                return ("related", None, None), "mb_same_rg"
            return ("unrelated", None, None), "mb_rg_disjoint"
        return ("unrelated", None, None), "mb_rg_distinct"

    for typ, a, b in zip(cand["type"], cand["entry_a"], cand["entry_b"]):
        a, b = int(a), int(b)
        key = (min(a, b), max(a, b))
        if key in seen or a in held_out or b in held_out:
            continue
        seen.add(key)
        res = track_label(a, b) if typ == "track" else other_label(typ, a, b)
        if res is None:
            continue
        (label, kind, derived), why = res
        if label == "unsure":
            continue
        add("mb", typ, sorted(pairs[a]), sorted(pairs[b]), label, kind,
            {"a": "left", "b": "right"}.get(derived), a, b, {"why": why})

    # ── provider relations (MB-asserted cover / remix / arrangement) ───────
    for ea, eb, kind, extra in con.execute("select entry_a, entry_b, kind, extra from entry_relation where origin='provider'"):
        key = (min(ea, eb), max(ea, eb))
        if key in seen or ea in held_out or eb in held_out:
            continue
        seen.add(key)
        derived = None
        try:
            d = json.loads(extra or "{}").get("derived_entry")
            derived = "left" if d == ea else "right" if d == eb else None
        except (ValueError, AttributeError):
            pass
        add("provider", types[ea], sorted(pairs[ea]), sorted(pairs[eb]), "related", kind, derived, ea, eb)

    # ── same-uploader video negatives ───────────────────────────────────────
    # Two different uploads by one channel are almost always different songs
    # ("X / artist (Cover)" vs "Y / artist (Cover)", "drum #1" vs "#2"): the
    # template-confusion negatives MB can't provide. Kept clean by requiring a
    # >3 s duration gap and no version-marker difference (cuts, previews,
    # live/3D versions of the same song are related, not unrelated).

    def uploader_set(eid):
        out = set()
        for p in pairs[eid]:
            if p[0] in ("youtube", "nicovideo"):
                for a, role, _ in lib_contrib.get(p, []):
                    if role == "uploader":
                        out.add(a)
        return out

    def video_only(eid):
        srcs = {s for s, i in pairs[eid] if f"{s}\x00{i}" in named}
        return bool(srcs) and srcs <= {"youtube", "nicovideo"}

    lib_contrib = defaultdict(list)
    for s, i, as_, ai, role in con.execute("select source, identifier, artist_source, artist_identifier, role from contribution where role='uploader'"):
        lib_contrib[(s, i)].append(((as_, ai), role, True))
    alias_names = defaultdict(list)
    for s, i, n in con.execute("select source, identifier, name from entry_alias"):
        alias_names[(s, i)].append(n)
    n_up = 0
    labeled = {(min(e["entry_a"], e["entry_b"]), max(e["entry_a"], e["entry_b"])) for e in examples}
    for typ, a, b in zip(cand["type"], cand["entry_a"], cand["entry_b"]):
        a, b = int(a), int(b)
        if typ != "track" or a in held_out or b in held_out or (min(a, b), max(a, b)) in labeled:
            continue
        if not (video_only(a) and video_only(b) and uploader_set(a) & uploader_set(b)):
            continue
        da = {d for p in pairs[a] for d in durations.get(p, ())}
        db = {d for p in pairs[b] for d in durations.get(p, ())}
        if not da or not db or min(abs(x - y) for x in da for y in db) <= 3000:
            continue
        ma, _ = F.markers([n for p in pairs[a] for n in alias_names[p]])
        mb, _ = F.markers([n for p in pairs[b] for n in alias_names[p]])
        if (ma ^ mb) - {"mv"}:
            continue
        labeled.add((min(a, b), max(a, b)))
        add("uploader_neg", "track", sorted(pairs[a]), sorted(pairs[b]), "unrelated", ea=a, eb=b, meta={"why": "same_uploader"})
        n_up += 1
    print(f"same-uploader negatives: {n_up}")

    # ── Jev teacher labels ──────────────────────────────────────────────────
    # Confident answers only (audit: >=0.9 tracks / >=0.8 others are 92-100%
    # right). Non-track prompts have no "related" answer: a release
    # different_identity that shares a release group is an edition (related);
    # a different_identity where the model said RELATE is skipped (can't tell
    # related from unrelated there).
    n_jev = defaultdict(int)
    if args.jev and Path(args.jev).exists():
        groups = defaultdict(set)
        child_entry = {}
        for s, i, eid in con.execute("select source, identifier, entry_id from entry_source"):
            child_entry[(s, i)] = eid
        for ps, pi, cs, ci in con.execute("select parent_source, parent_identifier, child_source, child_identifier from entry_child"):
            pe, ce = child_entry.get((ps, pi)), child_entry.get((cs, ci))
            if pe is not None and ce is not None and types.get(pe) == "release_group":
                groups[ce].add(pe)
        done = {(min(e["entry_a"], e["entry_b"]), max(e["entry_a"], e["entry_b"])) for e in examples}
        for line in Path(args.jev).read_text().splitlines():
            j = json.loads(line)
            a, b, typ = j["entry_a"], j["entry_b"], j["type"]
            k = (min(a, b), max(a, b))
            if k in done or a in held_out or b in held_out:
                continue
            min_conf = args.jev_min_conf_track if typ == "track" else args.jev_min_conf_other
            if j["confidence"] < min_conf or j["choice"] == "unsure":
                continue
            kind = derived = None
            if typ == "track":
                label = {"same_identity": "same", "derived": "related", "sibling": "sibling", "unrelated": "unrelated"}[j["choice"]]
                if label in ("related", "sibling"):
                    kind = j["kind"] if j.get("kind") not in (None, "not_applicable") else None
                    if j["choice"] == "derived":
                        derived = {"a_is_original": "right", "b_is_original": "left"}.get(j.get("direction"))
            else:
                if j["choice"] == "same_identity":
                    label = "same"
                elif typ == "release" and groups[a] & groups[b]:
                    label = "related"
                elif j["model_verdict"] == "RELATE":
                    continue
                else:
                    label = "unrelated"
            done.add(k)
            add("jev", typ, sorted(pairs[a]), sorted(pairs[b]), label, kind, derived, a, b,
                {"why": f"jev_{j['choice']}", "stratum": j["stratum"], "conf": j["confidence"]})
            n_jev[(typ, label)] += 1
    print(f"jev teacher examples: {dict(n_jev)}")

    # ── half-view negatives: same construction as probes, MB-unrelated ─────
    # Every probe entry gets up to PROBE_NEG_PER MB-labeled negatives built the
    # same way (its half vs the other entry's half, or the whole other entry
    # when that one can't be split), picked at random from its MB-labeled
    # candidates, so "both views are halves" never predicts the label.
    PROBE_NEG_PER = 3
    neg_by_entry = defaultdict(list)
    for e in examples:
        if e["tier"] == "mb" and e["label"] in ("unrelated", "related", "sibling"):
            neg_by_entry[e["entry_a"]].append(e)
            neg_by_entry[e["entry_b"]].append(e)
    for x, halves in probe_halves.items():
        cands = neg_by_entry.get(x, [])
        rng.shuffle(cands)
        for e in cands[:PROBE_NEG_PER]:
            y = e["entry_b"] if e["entry_a"] == x else e["entry_a"]
            hy = probe_halves.get(y)
            other = list(rng.choice(hy)) if hy else sorted(pairs[y])
            mine = list(rng.choice(halves))
            left, right = (mine, other) if rng.random() < 0.5 else (other, mine)
            derived = None
            if e["derived"]:
                derived_entry = e["entry_a"] if e["derived"] == "left" else e["entry_b"]
                derived = "left" if (derived_entry == x) == (left is mine) else "right"
            add("probe_neg", e["type"], left, right, e["label"], e["kind"], derived, ea=x, eb=y, meta=e["meta"])

    # Dev/test split by entity: hash of the smaller entry id's connected
    # component would be ideal; per-entry hash with "either side dev -> dev"
    # keeps one entity from straddling train and dev.
    def is_dev(eid):
        return int(hashlib.sha1(f"lm-{eid}".encode()).hexdigest(), 16) % 5 == 0
    for e in examples:
        e["split"] = "dev" if is_dev(e["entry_a"]) or is_dev(e["entry_b"]) else "train"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    df = pd.DataFrame([{k: e[k] for k in ("tier", "type", "label", "kind", "split")} | {"why": e["meta"].get("why")} for e in examples])
    print(f"{len(df)} examples -> {out}")
    print(df.groupby(["type", "tier", "label"]).size().to_string())
    print(df[df.type == "track"].groupby(["label", "kind"], dropna=False).size().to_string())
    print(df.groupby("why").size().to_string())
    print(df.groupby("split").size().to_string())


if __name__ == "__main__":
    main()
