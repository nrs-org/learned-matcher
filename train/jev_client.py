"""TypeSafe Jev client for the learned matcher: audit, defer-band judge, teacher.

`build_view` ports `src/pipeline/jev.rs::entry_view` (same fields, caps and
ordering) but works on any set of (source, identifier) pairs, so split
halves can be judged too. Prompt sets:

  v1  verbatim port of `jev.rs::build_questions` (what ships today)
  v2  aligned with the learned matcher's ontology (docs/plan-learned-matcher.md):
      full MV = same; cuts / TV size / other MV versions / live / remix /
      instrumental / cover / arrangement = derived; versions of one song where
      neither comes from the other = sibling; different songs = unrelated.

Every response is cached on disk by request hash: no pair is paid for twice.
"""

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from features import ROOT, Library

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
PRICE_PER_INPUT_TOKEN = 0.042 / 1e6
CACHE = ROOT / "data/learned-matcher/jev_cache"
VIDEO_SOURCES = ("youtube", "nicovideo", "soundcloud")

TOP_TITLES, CREDITED_NAMES_CAP, CHILD_TRACKS_CAP, TRACK_POSITIONS_CAP, HANDLES_CAP = 8, 8, 12, 5, 6


def api_key():
    if os.environ.get("TYPESAFE_API_KEY"):
        return os.environ["TYPESAFE_API_KEY"]
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("TYPESAFE_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("TYPESAFE_API_KEY not set")


# ── evidence view (port of jev.rs) ──────────────────────────────────────────

def extract_handle(identifier):
    if "://" not in identifier:
        return None
    after = identifier.split("://", 1)[1]
    core_path = re.split(r"[?#]", after)[0]
    segs = [s for s in core_path.split("/") if s]
    if not segs:
        return None
    core = segs[-1].lstrip("@")
    if not core:
        return None
    alldig = lambda s: bool(s) and s.isascii() and s.isdigit()
    if alldig(core) or alldig(core[2:] if core.startswith("id") else core):
        return None
    if len(core) == 24 and core.startswith("UC") and re.fullmatch(r"[A-Za-z0-9_-]+", core):
        return None
    if len(core) == 36 and core.count("-") == 4:
        return None
    if len(core) >= 20 and re.fullmatch(r"[0-9a-fA-F]+", core):
        return None
    if len(core) >= 5 and core[0].isascii() and core[0].isupper() and core[1:].isascii() and core[1:].isdigit():
        return None
    if len(core) >= 15 and re.search(r"\d", core) and re.search(r"[A-Z]", core) and re.search(r"[a-z]", core):
        return None
    try:
        decoded = urllib.parse.unquote(core, errors="strict")
    except UnicodeDecodeError:
        decoded = core
    return decoded or None


def entry_best_title(lib, eid):
    pairs = sorted(lib.entry_pairs.get(eid, []))
    sourced = []
    for p in pairs:
        for n, prim in lib.names.get(p, []):
            sourced.append((p[0], n, prim))
    if not sourced:
        return None
    sourced = sorted(set(sourced), key=lambda x: (x[0] in VIDEO_SOURCES, x[0], not x[2], x[1]))
    prim = [s for s in sourced if s[2]]
    return (prim or sourced)[0][1]


def build_view(lib: Library, pairs, typ):
    pairs = [tuple(p) for p in pairs]
    sourced = set()
    durations, dates, rtypes, ptypes = set(), set(), set(), set()
    peers, positions, children, groups = set(), set(), set(), set()
    for p in pairs:
        for n, prim in lib.names.get(p, []):
            sourced.add((p[0], n, prim))
        ds, rd, rt = lib.src.get(p, (set(), None, None))
        durations |= ds
        if rd:
            dates.add(rd)
        if rt:
            rtypes.add(rt)
        if typ == "artist":
            for item in lib.credited.get(p, []):
                ie = lib.entry_of.get(item)
                if ie is not None and lib.type_of.get(ie) == "track":
                    peers.add(ie)
        else:
            for a, _, _ in lib.contrib.get(p, []):
                ae = lib.entry_of.get(a)
                if ae is not None:
                    peers.add(ae)
        for parent, dn, tn in lib.parents.get(p, []):
            pe = lib.entry_of.get(parent)
            if pe is None:
                continue
            if lib.type_of.get(pe) == "release" and typ == "track":
                positions.add((pe, dn, tn))
            if lib.type_of.get(pe) == "release_group" and typ == "release":
                groups.add(pe)
        for k in lib.kids.get(p, []):
            ke = lib.entry_of.get(k)
            if ke is not None:
                children.add(ke)
    # canon_sourced_aliases: OR the primary flag per (source, name), clean sources first
    merged = {}
    for s, n, prim in sourced:
        merged[(s, n)] = merged.get((s, n), False) or prim
    ordered = sorted(merged.items(), key=lambda kv: (kv[0][0] in VIDEO_SOURCES, kv[0][0], not kv[1], kv[0][1]))
    credited = sorted({t for t in (entry_best_title(lib, e) for e in peers) if t})[:CREDITED_NAMES_CAP]
    rel_tracks = sorted({t for t in (entry_best_title(lib, e) for e in children if lib.type_of.get(e) == "track") if t})[:CHILD_TRACKS_CAP]
    pos = [{"release_title": entry_best_title(lib, r), "disc_no": d, "track_no": t}
           for r, d, t in sorted(positions, key=lambda x: (x[0], x[1] or 0, x[2] or 0))[:TRACK_POSITIONS_CAP]]
    handles = sorted({h for h in (extract_handle(i) for _, i in pairs) if h})[:HANDLES_CAP]
    return {
        "entry_type": typ,
        "titles": [{"source": s, "name": n} for (s, n), _ in ordered[:TOP_TITLES]],
        "durations_sec": [d / 1000 for d in sorted(durations)],
        "release_dates": sorted(dates),
        "release_types": sorted(rtypes),
        "primary_types": sorted(ptypes),
        "sources": sorted({s for s, _ in pairs}),
        "credited_names": credited,
        "track_positions": pos,
        "release_tracks": rel_tracks,
        "handles": handles,
        "release_group_ids": sorted(groups),
    }


# ── questions ───────────────────────────────────────────────────────────────

UNSURE = "the evidence is genuinely insufficient to decide either way."


def _choice(instructions, criteria):
    c = dict(criteria)
    c["unsure"] = UNSURE
    return {"type": "choice", "instructions": instructions, "criteria": c}


V1_IDENTITY = {
    "artist": _choice(
        "`a`/`b` are music-artist records aggregated from multiple databases (YouTube, Spotify, MusicBrainz, Discogs, etc.). Are they the SAME real-world person, group, or act? Language, romanization, capitalization, or channel-naming differences (e.g. a YouTube auto \"<name> - Topic\" channel) are NOT evidence of difference. A shared handle/username/official-link between the two records IS strong evidence for same_identity even when display names differ. Use different_identity when one is an individual member and the other the group/duo/unit they belong to (never merge a member with their group, even if closely associated), or when the shared name is generic/common with no other corroborating overlap (handle, associated work, official link).",
        {"same_identity": "same real-world person/group/act (different script/romanization/capitalization/channel-naming counts as this) -- merge.",
         "different_identity": "different entities -- e.g. a group vs. one of its members, or unrelated people/acts that merely share a name."}),
    "release": _choice(
        "`a`/`b` are release (album/EP/single/compilation) records from multiple databases, each listing `release_tracks`. Substantial tracklist overlap plus a matching or near-matching title/date is strong evidence FOR same_identity, even with catalog numbering/romanization/minor track-order differences (the same release often gets catalogued independently by several providers). Evidence FOR different_identity: a materially different tracklist/edition (e.g. a Deluxe/Anniversary edition, a live/remix album, a separate regional release with different content), release dates far apart with no edition link, or a generic shared title (e.g. \"Various Artists\") that could label unrelated releases. OVERRIDE: if `a` and `b` share a `release_group_ids` value, treat that as strong evidence FOR different_identity instead -- MusicBrainz/Discogs deliberately split one release_group into separate release entities to represent distinct editions/pressings, so a shared group id means \"different edition\" far more often than \"duplicate catalogued twice\", even when title/tracklist look alike.",
        {"same_identity": "same release (overlapping tracklist, matching title/date) catalogued independently by different providers -- merge.",
         "different_identity": "different releases -- a materially different tracklist/edition, a shared release_group id (distinct editions), unrelated dates, or a generic shared title with no real tracklist overlap."}),
    "release_group": _choice(
        "`a`/`b` are release-group records -- the overarching creative work (e.g. \"the album\") independent of any specific edition, pressing, or regional release. Are they the SAME overarching work? Unlike at the `release` level, differences in pressing, edition, bonus-track count, regional tracklist, or title language/romanization do NOT matter here -- all still the same release_group. Use different_identity only when the underlying creative work itself differs -- a distinct album/EP project, an unrelated work sharing a generic title, or a compilation vs. the original work it draws from.",
        {"same_identity": "same overarching creative work, regardless of edition/pressing differences between member releases -- merge.",
         "different_identity": "different creative works -- not merely a different edition of the same one."}),
    "track": _choice(
        "`a`/`b` are track records from multiple databases. Classify the relationship: same_identity (the same recording -- merge), related_variant (a different mix/arrangement/component of the same song -- e.g. Instrumental, Off Vocal, Karaoke, Acapella, Remix, Arrange/Arrangement, a named lineup/event version -- keep separate but linked), or unrelated (a different song, or an independent performer's own separate recording sharing only the title -- no real connection). Language, romanization, capitalization, an edition suffix like \"(TV size)\", or an appended cover-credit tag (e.g. \"<title> ／ <performer>(Cover)\", a common YouTube convention) are NOT evidence against same_identity. A suffix naming a different mix/arrangement/component means NOT same_identity, but IS related_variant, not unrelated -- don't collapse that distinction. When base titles are effectively identical (ignoring punctuation like full/half-width comma) with no version/arrangement suffix, default to same_identity even without a duration match: a music-video cut commonly runs 10-30s longer than an audio cut of the same song, so that gap alone isn't contradicting evidence -- and neither is partial credited-artist overlap, since source data is often incomplete. EXCEPTION: a short generic label reused across releases (numbered MC/talk segments, \"Intro\", \"Outro\", \"Encore\") is weak evidence even on an exact match, since each event has its own distinct segment under the same name -- require other corroboration, defaulting to unrelated or unsure without it.",
        {"same_identity": "the same recording (the same actual audio content) -- merge.",
         "related_variant": "a different version/mix/arrangement/component of the same song (e.g. instrumental vs. vocal, a named arrangement/lineup) -- related, not merged.",
         "unrelated": "no real connection -- a different song, or an independent performer's own recording sharing only the title."}),
}
V1_NOULS = {
    "title_match": {"type": "noul", "instructions": "Do a's and b's titles refer to the same underlying work, ignoring language/romanization variants, formatting, and edition/version suffixes?"},
    "duration_consistent": {"type": "noul", "instructions": "Are a's and b's reported durations consistent with being the same recording, allowing for missing data on either side and normal encoding/rounding variance (a few seconds)? If either side has no duration data, treat that as not contradicting a match."},
}

V2_TRACK = {
    "type": "choice",
    "instructions": {
        "task": "`a` and `b` are two track records for a music library, each aggregated from providers (YouTube, Spotify, MusicBrainz, Discogs, ...). Decide how the two recordings relate.",
        "same_identity_means": [
            "the same recording catalogued twice: same performers, same performance, same song length (encoding, rounding and a few seconds of silence don't matter)",
            "a FULL music video of the song and its audio release: an official MV can stand in for the audio track, so it counts as the same even when the video runs 10-60 s longer for an intro/outro",
            "titles that differ only by language, romanization, punctuation, or upload decoration (【MV】, 'Official Video', '/ artist', '(Cover)' credit tags, hashtags)",
        ],
        "not_same": [
            "a shortened version of the song: short ver., TV size, game size, cut, one chorus, #shorts, preview/teaser/crossfade",
            "an alternate MV version (dance shot ver., close-up ver., other named video versions) — it can't be guaranteed to replace the audio",
            "a live performance vs the studio recording, a remix, an instrumental / off vocal / karaoke / a cappella version, an arrangement, a re-recording, a named lineup or event version",
            "the same song recorded by a different performer (a cover)",
            "two different named versions of one song, e.g. '(2022 3期生 ver.)' vs '(2022 ゲーマーズ ver.)' — they are siblings, not the same",
            "stems / multitrack parts of a song ('(Strings_Stem)', '(Drum_Stem)'): each part is derived from the song, and two different parts are siblings",
            "a recording from a specific concert or show ('<昼公演>', 'day 2', 'live at ...') vs the studio song is live, i.e. derived",
        ],
        "unrelated_means": "different songs, even by the same artist, from the same channel, or wrapped in the same title template (compare the song name itself, not the shared '(Instrumental)' / '/ artist' / 【MV】 wrapping); generic recurring labels (MC1 vs MC2, Intro, Outro, numbered drum/talk segments) with different numbers or no corroboration",
        "placeholders": "a title like 'Private video' or 'Deleted video' carries no information about the song; with nothing else linking the two, answer unsure",
        "evidence": "durations_sec are lists of reported lengths; a gap of more than ~15 s between the closest pair, without an MV intro/outro explanation, argues against same_identity. credited_names may be incomplete on either side.",
    },
    "criteria": {
        "same_identity": "the same recording, or a full MV of it — merge",
        "derived": "one is a version made from the other: a short/cut/TV-size version, an alternate MV version, live, remix, instrumental/off vocal/karaoke, a cappella, arrangement, or a cover of the other",
        "sibling": "both are different versions of the same song but neither is made from the other (e.g. two different performers' covers, or an acoustic ver. and a lineup ver.)",
        "unrelated": "different songs, or no real connection",
        "unsure": UNSURE,
    },
}
V2_KIND = {
    "type": "choice",
    "instructions": "Premise: assume `a` and `b` are two versions of the same song (if they aren't, answer not_applicable). What distinguishes them?",
    "criteria": {
        "edit": "one is a shortened version: short ver., TV size, game size, cut, #shorts, preview, or an alternate MV cut",
        "live": "a live performance",
        "remix": "a remix",
        "instrumental": "instrumental, off vocal, karaoke or a cappella",
        "cover": "a different performer's recording of the song",
        "arrangement": "a re-arrangement (acoustic, orchestral, piano, named arrange)",
        "alt_version": "another named version: lineup/unit version, language version, re-recording, anniversary ver.",
        "not_applicable": "they aren't versions of the same song, or they are the same recording",
    },
}
V2_DIRECTION = {
    "type": "choice",
    "instructions": "Premise: assume `a` and `b` are two versions of the same song. Is one of them the original that the other was made from (the full studio/original release vs. its cut, live, remix, instrumental, cover, ...)?",
    "criteria": {
        "a_is_original": "`b` was made from `a`",
        "b_is_original": "`a` was made from `b`",
        "neither": "neither is the source of the other, or they are the same recording, or they aren't versions of one song",
    },
}


def questions(typ, version):
    if version == "v2.1":
        version = "v2"  # v2.1 only edits V2_TRACK in place; kept as a label for reports
    if version == "v1":
        return {"identity": V1_IDENTITY.get(typ, V1_IDENTITY["track"]), **V1_NOULS}
    if version == "v2":
        if typ == "track":
            return {"identity": V2_TRACK, "kind": V2_KIND, "direction": V2_DIRECTION}
        return {"identity": V1_IDENTITY[typ]}
    raise ValueError(version)


# ── transport ───────────────────────────────────────────────────────────────

class Jev:
    def __init__(self, concurrency=12):
        self.key = api_key()
        self.concurrency = concurrency
        CACHE.mkdir(parents=True, exist_ok=True)
        self.spent_tokens = 0
        self.calls = 0
        self.cache_hits = 0

    def _post(self, body):
        h = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        path = CACHE / f"{h}.json"
        if path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text())
        data = json.dumps(body, ensure_ascii=False).encode()
        for attempt in range(6):
            req = urllib.request.Request(API_URL, data=data, method="POST", headers={
                "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    resp = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 529, 500, 502, 503) and attempt < 5:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"TypeSafe {e.code}: {e.read()[:300]!r}")
            except (urllib.error.URLError, TimeoutError):
                if attempt < 5:
                    time.sleep(2 ** attempt)
                    continue
                raise
        path.write_text(json.dumps(resp, ensure_ascii=False))
        self.calls += 1
        self.spent_tokens += resp.get("usage", {}).get("input_tokens", 0)
        return resp

    def ask(self, view_a, view_b, typ, version):
        body = {"state": {"a": view_a, "b": view_b}, "model": MODEL, "questions": questions(typ, version)}
        return self._post(body)

    def ask_many(self, jobs):
        """jobs: list of (view_a, view_b, typ, version) -> list of responses (None on error)."""
        def one(j):
            try:
                return self.ask(*j)
            except Exception as e:  # keep the batch going; report errors
                return {"error": str(e)}
        with ThreadPoolExecutor(self.concurrency) as ex:
            return list(ex.map(one, jobs))

    def cost(self):
        return self.spent_tokens * PRICE_PER_INPUT_TOKEN
