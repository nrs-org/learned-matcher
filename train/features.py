"""Pair features for the learned matcher.

A *view* is any set of (source, identifier) pairs from the snapshot: a whole
entry, or one split half of it. `Library.view()` folds per-pair facts into a
view, `pair_features()` compares two views.

Deliberately absent (they would leak how the training data was built):
provider overlap / per-side source lists, number of sources or aliases, shared
identifiers. Split-probe positives have disjoint providers by construction;
real duplicates don't.

Encoder: gbnam8/jp-music-title-encoder (v7-L6, ONNX), `title [A] artist` for
tracks, raw names otherwise, truncated to 256 dims and re-normalized.
"""

import json
import re
import sqlite3
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from paths import DATA

ENC_DIR = DATA / "learned-matcher/encoder"
VIDEO_SOURCES = {"youtube", "nicovideo", "soundcloud", "bilibili"}
DIM = 256
PAIR_DIMS = 64  # dims of |u-v| and u*v fed to the head

MARKERS = {
    "live": r"\blive\b|ライブ|生歌",
    "remix": r"remix|リミックス|\bmix\)|bootleg",
    "instrumental": r"instrumental|\binst\b|inst\.|インスト|off ?vocal|karaoke|カラオケ|backing track",
    "acoustic": r"acoustic|アコースティック|unplugged",
    "cover": r"\bcover\b|カバー|歌ってみた|歌わせていただきました",
    "medley": r"medley|メドレー",
    "short": r"tv ?size|short ?ver|short version|cut ?ver|one chorus|ショート|tv ?ver|\bshort\b|#shorts|試聴|preview|teaser|crossfade|クロスフェード",
    "mv": r"\bmv\b|music video|\bpv\b|ミュージックビデオ|official video",
    "arrange": r"arrange|アレンジ|編曲",
    "edit": r"\bedit\b|radio edit",
    "remaster": r"remaster|リマスター",
    "ver": r"\bver\b|ver\.|version|バージョン",
    "acappella": r"a ?cappella|アカペラ",
    "full": r"full ?ver|full size|フル",
    "demo": r"\bdemo\b",
    "stem": r"stem\b|multitrack|ステム|パラデータ",
    "performance": r"昼公演|夜公演|\bday ?\d|\bnight\b|公演|fes\b",
}
MARKER_RES = {k: re.compile(v, re.I) for k, v in MARKERS.items()}
HOLO = re.compile(r"hololive|ホロライブ", re.I)
NAMED_VER = re.compile(r"([\w぀-ヿ一-鿿]+)\s*ver(?:sion|\.)?", re.I)
DIGITS = re.compile(r"\d+")
PLACEHOLDER = re.compile(r"^\[?(private video|deleted video|untitled|unknown)\]?$", re.I)
BRACKETS = re.compile(r"[\(（\[【<＜〈《][^\)）\]】>＞〉》]*[\)）\]】>＞〉》]")
SLASH_TAIL = re.compile(r"\s*[/／|｜]\s*[^/／|｜]*$")
ARTIST_STRIP = [re.compile(x, re.I) for x in (
    r"\s*-\s*topic$", r"\s*\((?:cv|c\.v|vo|voice)[.:：]?[^)]*\)", r"\s*（(?:cv|vo)[.:：]?[^）]*）",
    r"\s*\(\d+\)$", r"\s*\(all\)$", r"様$", r"\s*official$", r"\s*公式$", r"\s*vevo$",
    r"\s*\bch\..*$", r"\s*ch\.?$", r"\s*channel$", r"\s*チャンネル$")]
GENERIC_ARTIST = re.compile(r"^(release|various artists|ヴァリアス・アーティスト|v\.?a\.?|unknown artist|不明|anonymous|traditional)$", re.I)
CJK = re.compile(r"[぀-ヿ一-鿿ｦ-ﾟ]")


def norm(s):
    s = unicodedata.normalize("NFKC", s).casefold()
    return re.sub(r"\s+", " ", s).strip()


def trigrams(s):
    s = re.sub(r"[\s\W_]+", "", s)
    return {s[i:i + 3] for i in range(max(1, len(s) - 2))} if s else set()


def tokens(s):
    return set(t for t in re.split(r"[\s\W_]+", s) if t)


def jacc(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class View:
    type: str
    names: list = field(default_factory=list)       # (source, name)
    texts: list = field(default_factory=list)       # encoder input per name (conditioned)
    durations: set = field(default_factory=set)
    artists: set = field(default_factory=set)       # artist entry ids
    artist_names: set = field(default_factory=set)
    releases: set = field(default_factory=set)      # parent release entry ids
    positions: set = field(default_factory=set)     # (release eid, disc, track)
    children: set = field(default_factory=set)      # child entry ids (release/rg)
    credits: set = field(default_factory=set)       # artist: credited item entry ids
    groups: set = field(default_factory=set)        # release: parent release-group entry ids
    child_titles: list = field(default_factory=list)  # release: (disc, track, core title) of tracks
    uploaders: set = field(default_factory=set)     # video uploader artist entry ids
    years: set = field(default_factory=set)
    rtypes: set = field(default_factory=set)
    rec_years: set = field(default_factory=set)     # MB: first-release years of this view's recordings
    work_first: set = field(default_factory=set)    # MB: first year any recording of its works came out
    video_only: bool = False
    has_video: bool = False


class Library:
    def __init__(self, db):
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        self.entry_of, self.type_of = {}, {}
        for eid, t in con.execute("select id, entry_type from entry"):
            self.type_of[eid] = t
        self.src = {}
        for s, i, eid, d, dall, rd, rt, pt in con.execute(
                "select source, identifier, entry_id, duration_ms, duration_ms_all, release_date, release_type, primary_type from entry_source"):
            p = (s, i)
            self.entry_of[p] = eid
            ds = {d} if d else set()
            if dall:
                try:
                    ds.update(json.loads(dall))
                except ValueError:
                    pass
            self.src[p] = (ds, rd, rt or pt)
        self.names = defaultdict(list)
        for s, i, n, prim in con.execute('select source, identifier, name, "primary" from entry_alias order by "primary" desc, id'):
            self.names[(s, i)].append((n, bool(prim)))
        self.contrib = defaultdict(list)   # item pair -> [(artist pair, role, main)]
        self.credited = defaultdict(list)  # artist pair -> [item pair]
        for s, i, as_, ai, role, main in con.execute(
                "select source, identifier, artist_source, artist_identifier, role, main_artist from contribution order by id"):
            self.contrib[(s, i)].append(((as_, ai), role, bool(main)))
            self.credited[(as_, ai)].append((s, i))
        self.parents = defaultdict(list)
        self.kids = defaultdict(list)
        for ps, pi, cs, ci, dn, tn in con.execute(
                "select parent_source, parent_identifier, child_source, child_identifier, disc_no, track_no from entry_child"):
            self.parents[(cs, ci)].append(((ps, pi), dn, tn))
            self.kids[(ps, pi)].append((cs, ci))
        self.entry_pairs = defaultdict(list)
        for p, eid in self.entry_of.items():
            self.entry_pairs[eid].append(p)
        # optional MB year facts (mb_years.sql): who recorded a song first
        mb = DATA / "learned-matcher/mb"
        self.rec_year, self.rec_works, self.work_year = {}, defaultdict(set), {}
        if (mb / "rec_year.tsv").exists():
            for line in (mb / "rec_year.tsv").read_text().splitlines():
                g, y = line.split("\t")
                self.rec_year[g] = int(y)
            for line in (mb / "work_year.tsv").read_text().splitlines():
                g, y = line.split("\t")
                self.work_year[g] = int(y)
            for line in (mb / "rec_work.tsv").read_text().splitlines():
                parts = line.split("\t")
                self.rec_works[parts[0]].add(parts[1])

    def best_name(self, pair):
        ns = self.names.get(pair)
        if ns:
            return ns[0][0]
        eid = self.entry_of.get(pair)
        for p in self.entry_pairs.get(eid, []):
            if self.names.get(p):
                return self.names[p][0][0]
        return None

    def primary_artist(self, pair):
        cs = self.contrib.get(pair, [])
        for a, role, main in cs:
            if main and role == "listed_artist":
                return a
        for a, role, main in cs:
            if role == "uploader":
                return a
        return None

    def view(self, pairs, typ):
        v = View(type=typ)
        named_sources = set()
        for p in map(tuple, pairs):
            ds, rd, rt = self.src.get(p, (set(), None, None))
            v.durations |= ds
            if rd and rd[:4].isdigit():
                v.years.add(int(rd[:4]))
            if rt:
                v.rtypes.add(rt.lower())
            pa = self.primary_artist(p) if typ == "track" else None
            pa_name = self.best_name(pa) if pa else None
            for n, _ in self.names.get(p, []):
                v.names.append((p[0], n))
                v.texts.append(f"{n} [A] {pa_name}" if pa_name else n)
                named_sources.add(p[0])
            for a, role, main in self.contrib.get(p, []):
                if main or role in ("listed_artist", "uploader", "vocal"):
                    ae = self.entry_of.get(a)
                    if ae is not None:
                        v.artists.add(ae)
                    an = self.best_name(a)
                    if an:
                        v.artist_names.add(an)
            for a, role, main in self.contrib.get(p, []):
                if role == "uploader" and self.entry_of.get(a) is not None:
                    v.uploaders.add(self.entry_of[a])
            for parent, dn, tn in self.parents.get(p, []):
                pe = self.entry_of.get(parent)
                if pe is not None and typ == "release" and self.type_of.get(pe) == "release_group":
                    v.groups.add(pe)
                if pe is not None and self.type_of.get(pe) == "release":
                    v.releases.add(pe)
                    if tn is not None:
                        v.positions.add((pe, dn or 1, tn))
            for k in self.kids.get(p, []):
                ke = self.entry_of.get(k)
                if ke is not None and self.type_of.get(ke) in ("track", "release"):
                    v.children.add(ke)
                if typ == "release" and self.names.get(k):
                    v.child_titles.append(core_title(self.names[k][0][0]))
            if typ == "track" and p[0] == "musicbrainz" and "/recording/" in p[1]:
                gid = p[1].rstrip("/").rsplit("/", 1)[-1]
                if gid in self.rec_year:
                    v.rec_years.add(self.rec_year[gid])
                v.work_first |= {self.work_year[w] for w in self.rec_works.get(gid, ()) if w in self.work_year}
            if typ == "artist":
                for item in self.credited.get(p, [])[:500]:
                    ie = self.entry_of.get(item)
                    if ie is not None:
                        v.credits.add(ie)
        # dedupe names, keep order
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


class Encoder:
    def __init__(self, threads=16):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(ENC_DIR / "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(ENC_DIR / "tokenizer.json"))
        self.tok.enable_truncation(48)
        self.cache = {}

    def embed(self, texts, batch=128):
        todo = sorted({t for t in texts if t not in self.cache}, key=len)
        for k in range(0, len(todo), batch):
            chunk = todo[k:k + batch]
            enc = self.tok.encode_batch(chunk)
            L = max(len(e.ids) for e in enc)
            ids = np.zeros((len(chunk), L), dtype=np.int64)
            mask = np.zeros((len(chunk), L), dtype=np.int64)
            # tokenizer.json pads to the batch's longest: the mask must come
            # from the encoding, or pad tokens get attended and mean-pooled
            # and a text's vector depends on its batch neighbours.
            for r, e in enumerate(enc):
                ids[r, :len(e.ids)] = e.ids
                mask[r, :len(e.ids)] = e.attention_mask
            out = self.sess.run(["embedding"], {"input_ids": ids, "attention_mask": mask})[0][:, :DIM]
            out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
            for t, vec in zip(chunk, out.astype(np.float32)):
                self.cache[t] = vec
        return self.cache

    def save(self, path):
        keys = list(self.cache)
        np.save(path, np.stack([self.cache[k] for k in keys]) if keys else np.zeros((0, DIM), np.float32))
        Path(str(path) + ".keys.json").write_text(json.dumps(keys, ensure_ascii=False))

    def load(self, path):
        p = Path(path)
        if p.exists():
            keys = json.loads(Path(str(path) + ".keys.json").read_text())
            mat = np.load(p)
            self.cache.update(zip(keys, mat))


def core_title(name):
    """Title with bracketed segments, a trailing '/ artist' part and version
    markers removed: what's left should name the song itself."""
    t = HOLO.sub("", norm(name))
    stripped = BRACKETS.sub(" ", t)
    if stripped.strip():
        t = stripped
    tail = SLASH_TAIL.sub("", t)
    if tail.strip():
        t = tail
    for r in MARKER_RES.values():
        t = r.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip(" -_~・")


def artist_core(name):
    """Artist / channel name without platform wrappers: '- Topic', '(CV. …)',
    Discogs '(n)', SoundCloud '(All)', 'Official', '… Ch.'."""
    t = norm(name)
    for r in ARTIST_STRIP:
        new = r.sub("", t).strip()
        if new:
            t = new
    return re.sub(r"[\s・._-]+", "", t)


def artist_core_raw(name):
    t = norm(name)
    for r in ARTIST_STRIP:
        new = r.sub("", t).strip()
        if new:
            t = new
    return t


def bracket_contents(name):
    return {re.sub(r"\s+", " ", m.group(0)[1:-1]).strip() for m in BRACKETS.finditer(norm(name))} - {""}


def markers(names):
    found, named = set(), set()
    for n in names:
        t = HOLO.sub("", norm(n))
        for k, r in MARKER_RES.items():
            if r.search(t):
                found.add(k)
        for m in NAMED_VER.finditer(t):
            named.add(m.group(1))
    return found, named


def mat(cache, texts):
    if not texts:
        return np.zeros((0, DIM), np.float32)
    return np.stack([cache[t] for t in texts])


def cos(A, B):
    """Cosine matrix, defined so any implementation gets the same float32:
    normalise and multiply in float64, then round. float32 BLAS sums in a
    kernel-specific order, and the trees split at ulp level near 1.0
    (identical texts), so float32 dot products are not reproducible."""
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    return (A @ B.T).astype(np.float32)


def meanbest(S):
    return float((S.max(1).astype(np.float64).mean() + S.max(0).astype(np.float64).mean()) / 2)


FEATURE_NAMES = None


def pair_features(a: View, b: View, cache):
    f = {}
    na = [n for _, n in a.names]
    nb = [n for _, n in b.names]
    A, B = mat(cache, a.texts), mat(cache, b.texts)
    if len(A) and len(B):
        S = cos(A, B)
        i, j = np.unravel_index(np.argmax(S), S.shape)
        f["enc_max"] = float(S[i, j])
        f["enc_meanbest"] = meanbest(S)
        f["enc_minbest"] = float(min(S.max(1).min(), S.max(0).min()))
        u, v = A[i, :PAIR_DIMS], B[j, :PAIR_DIMS]
        diff, prod = np.abs(u - v), u * v
        best = (na[i], nb[j])
        if a.type == "track":
            TA, TB = mat(cache, na), mat(cache, nb)
            T = cos(TA, TB)
            f["enc_title_max"] = float(T.max())
            f["enc_title_meanbest"] = meanbest(T)
        else:
            f["enc_title_max"], f["enc_title_meanbest"] = f["enc_max"], f["enc_meanbest"]
    else:
        f.update(enc_max=np.nan, enc_meanbest=np.nan, enc_minbest=np.nan, enc_title_max=np.nan, enc_title_meanbest=np.nan)
        diff = prod = np.full(PAIR_DIMS, np.nan, np.float32)
        best = (na[0] if na else "", nb[0] if nb else "")
    for k in range(PAIR_DIMS):
        f[f"d{k}"] = float(diff[k])
        f[f"p{k}"] = float(prod[k])

    # artist names (tracks/releases): encoder cosine between credited artists
    if a.artist_names and b.artist_names:
        AA, AB = mat(cache, sorted(a.artist_names)), mat(cache, sorted(b.artist_names))
        f["artist_enc_max"] = float(cos(AA, AB).max())
    else:
        f["artist_enc_max"] = np.nan

    # lexical
    nna, nnb = {norm(x) for x in na}, {norm(x) for x in nb}
    f["lex_exact"] = float(bool(nna & nnb))
    ta = [trigrams(x) for x in nna]
    tb = [trigrams(x) for x in nnb]
    f["lex_tri_max"] = max((jacc(x, y) for x in ta for y in tb), default=np.nan)
    f["lex_tok_max"] = max((jacc(tokens(x), tokens(y)) for x in nna for y in nnb), default=np.nan)
    ba, bb = norm(best[0]), norm(best[1])
    f["len_ratio"] = min(len(ba), len(bb)) / max(len(ba), len(bb), 1)
    da, db = set(DIGITS.findall(ba)), set(DIGITS.findall(bb))
    f["num_conflict"] = float(bool(da) and bool(db) and da != db)
    f["num_xor"] = float(bool(da) != bool(db))
    ca = sum(bool(CJK.search(x)) for x in na) / max(len(na), 1)
    cb = sum(bool(CJK.search(x)) for x in nb) / max(len(nb), 1)
    f["cjk_min"], f["cjk_absdiff"] = min(ca, cb), abs(ca - cb)

    # core titles (brackets / '/ artist' / markers stripped) and bracket contents
    ca_ = {core_title(x) for x in na} - {""}
    cb_ = {core_title(x) for x in nb} - {""}
    f["core_exact"] = float(bool(ca_ & cb_))
    f["core_tri_max"] = max((jacc(trigrams(x), trigrams(y)) for x in ca_ for y in cb_), default=np.nan)
    pa_ = set().union(*[bracket_contents(x) for x in na]) if na else set()
    pb_ = set().union(*[bracket_contents(x) for x in nb]) if nb else set()
    f["bracket_jacc"] = jacc(pa_, pb_) if pa_ and pb_ else np.nan
    f["bracket_xor"] = float(bool(pa_) != bool(pb_))
    if a.type == "artist":
        aca = {artist_core(x) for x in na} - {""}
        acb = {artist_core(x) for x in nb} - {""}
        f["artist_core_exact"] = float(bool(aca & acb))
        f["artist_core_tri_max"] = max((jacc(trigrams(x), trigrams(y)) for x in aca for y in acb), default=np.nan)
        f["generic_name"] = float(any(GENERIC_ARTIST.match(norm(artist_core_raw(x))) for x in na + nb))
    else:
        f["artist_core_exact"] = f["artist_core_tri_max"] = f["generic_name"] = np.nan
    if a.artist_names and b.artist_names:
        f["credit_core_overlap"] = float(bool({artist_core(x) for x in a.artist_names} & {artist_core(x) for x in b.artist_names}))
    else:
        f["credit_core_overlap"] = np.nan
    f["placeholder"] = float(any(PLACEHOLDER.match(norm(x)) for x in na) or any(PLACEHOLDER.match(norm(x)) for x in nb))

    # version markers over the full alias sets
    ma, va = markers(na)
    mb, vb = markers(nb)
    for k in MARKERS:
        f[f"m_xor_{k}"] = float((k in ma) != (k in mb))
        f[f"m_both_{k}"] = float(k in ma and k in mb)
    f["m_xor_count"] = float(len(ma ^ mb))
    # Signed (antisymmetric) features, "s_" prefix: only the relation head's
    # direction model reads them; the identity model must stay symmetric.
    for k in MARKERS:
        f[f"s_m_{k}"] = float(k in ma) - float(k in mb)
    f["s_named_ver"] = float(bool(va)) - float(bool(vb))
    f["s_video"] = float(a.video_only) - float(b.video_only)
    f["s_dur"] = ((max(a.durations) - max(b.durations)) / 1000) if a.durations and b.durations else np.nan
    f["s_len"] = float(len(ba) - len(bb))
    f["s_names"] = float(len(na) - len(nb))
    f["s_artists"] = float(len(a.artists) - len(b.artists))
    f["named_ver_conflict"] = float(bool(va) and bool(vb) and not (va & vb))
    f["named_ver_xor"] = float(bool(va) != bool(vb))

    # duration
    if a.durations and b.durations:
        deltas = [abs(x - y) for x in a.durations for y in b.durations]
        d = min(deltas)
        f["dur_known"] = 1.0
        f["dur_delta_s"] = d / 1000
        f["dur_rel"] = d / max(max(a.durations), max(b.durations), 1)
        f["dur_max_delta_s"] = (max(max(a.durations), max(b.durations)) - min(min(a.durations), min(b.durations))) / 1000
    else:
        f.update(dur_known=0.0, dur_delta_s=np.nan, dur_rel=np.nan, dur_max_delta_s=np.nan)
    # originality: years between a side's first release and the first
    # recording of its song (MB works). 0 = the original; > 0 = a later
    # version. Both > 0 suggests two later versions of one song (siblings).
    def gap(v):
        if v.rec_years and v.work_first:
            return max(0, min(v.rec_years) - min(v.work_first))
        return np.nan
    ga, gb = gap(a), gap(b)
    f["orig_gap_min"] = np.nanmin([ga, gb]) if not (np.isnan(ga) and np.isnan(gb)) else np.nan
    f["orig_gap_max"] = np.nanmax([ga, gb]) if not (np.isnan(ga) and np.isnan(gb)) else np.nan
    f["orig_known"] = float(not np.isnan(ga)) + float(not np.isnan(gb))
    f["s_orig_gap"] = (ga - gb) if not (np.isnan(ga) or np.isnan(gb)) else np.nan
    f["video_sides"] = float(a.video_only) + float(b.video_only)
    # video side longer than the other (MV intro/outro) vs shorter (cut)
    if a.durations and b.durations and a.video_only != b.video_only:
        vid, aud = (a, b) if a.video_only else (b, a)
        f["mv_minus_audio_s"] = (max(vid.durations) - max(aud.durations)) / 1000
    else:
        f["mv_minus_audio_s"] = np.nan

    # credits / structure
    f["artists_known"] = float(bool(a.artists) and bool(b.artists))
    f["artist_jacc"] = jacc(a.artists, b.artists)
    f["artist_shared"] = float(bool(a.artists & b.artists))
    f["same_release"] = float(bool(a.releases & b.releases))
    f["same_position"] = float(bool(a.positions & b.positions))
    shared_rel = {p[0] for p in a.positions} & {p[0] for p in b.positions}
    f["position_conflict"] = float(bool(shared_rel) and not (a.positions & b.positions))
    f["groups_known"] = float(bool(a.groups) and bool(b.groups))
    f["same_group"] = float(bool(a.groups & b.groups))
    f["group_conflict"] = float(bool(a.groups) and bool(b.groups) and not (a.groups & b.groups))
    f["same_uploader"] = float(bool(a.uploaders & b.uploaders))
    # title-level tracklist comparison: works even when the two releases'
    # tracks were never merged into shared entries
    if a.child_titles and b.child_titles:
        ta_, tb_ = set(a.child_titles) - {""}, set(b.child_titles) - {""}
        f["tracklist_title_jacc"] = jacc(ta_, tb_)
        f["tracklist_title_cover"] = len(ta_ & tb_) / max(min(len(ta_), len(tb_)), 1)
        f["tracklist_len_ratio"] = min(len(ta_), len(tb_)) / max(len(ta_), len(tb_), 1)
        f["tracklist_extra"] = float(abs(len(ta_) - len(tb_)))
    else:
        f["tracklist_title_jacc"] = f["tracklist_title_cover"] = f["tracklist_len_ratio"] = f["tracklist_extra"] = np.nan
    f["children_known"] = float(bool(a.children) and bool(b.children))
    f["children_jacc"] = jacc(a.children, b.children)
    f["children_ratio"] = (min(len(a.children), len(b.children)) / max(len(a.children), len(b.children))) if a.children and b.children else np.nan
    f["credits_jacc"] = jacc(a.credits, b.credits)
    f["credits_shared_log"] = float(np.log1p(len(a.credits & b.credits)))
    f["years_known"] = float(bool(a.years) and bool(b.years))
    f["year_delta"] = min((abs(x - y) for x in a.years for y in b.years), default=np.nan)
    f["rtype_equal"] = float(bool(a.rtypes & b.rtypes)) if a.rtypes and b.rtypes else np.nan
    f["type_id"] = float(["track", "artist", "release", "release_group"].index(a.type))
    return f
