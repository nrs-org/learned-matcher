# learned-matcher

Learned soft-dedup for [musiclib-rs](https://github.com/nrs-org/musiclib-rs):
the pair classifier's training pipeline, its runtime, and the match script
that plugs it into musiclib-rs's `softmatch`.

| Path | What |
|---|---|
| `train/` | Python training pipeline (uv). Plan and results: `docs/plan-learned-matcher.md`. |
| `inference/` | `libinference.so` cdylib: the matcher (`matcher`), the TypeSafe client (`typesafe`), optional Vulkan title encoder (`vulkan`). |
| `rhai/` | `match.learned.rhai` (policy), `jev.rhai` + `jev/*.json` (Jev refine step), `csv.rhai`. |
| `docs/` | Design docs: `plan-learned-matcher.md`, `plan-v15-runtime.md`, `plan-jev-in-script.md`. |

musiclib-rs owns the host half of the contract: the generic `ffi` Rhai module
(build it with `--features ffi`), the lazy `Entry` type, `pair_facts_json`
(`musiclib-pair-facts/1`, `src/pipeline/pair_facts.rs`) and the
`decide`/`refine` hooks. Its tests for `jev.rhai`, `csv.rhai`, the cdylib ABI
and the pair-facts golden file look for this repo at `../learned-matcher`.

## Data

The pipeline reads and writes musiclib-rs's gitignored `data/` directory (DB
snapshots, gold sets, models, bundles, parity fixtures). It expects
musiclib-rs checked out next to this repo; set `MUSICLIB_DATA` to point
elsewhere (`train/paths.py`; the parity test reads it too).

## Runtime

```bash
nix develop                                   # or: direnv with `use flake`
cargo build -p inference --release --lib      # add `--features vulkan` for the GPU encoder
(cd train && uv run python bundle.py)         # → $MUSICLIB_DATA/learned-matcher/bundle/v17
cargo test -p inference --release --features matcher --test matcher_parity -- --nocapture
```

Install for musiclib-rs (`softmatch` reads `<config_dir>/match.rhai`; modules
and `ffi::open` paths resolve relative to that file's directory):

```bash
cd ~/.config/musiclib-rs
for f in jev.rhai jev csv.rhai; do ln -sfn ~/dev/nrs-org/learned-matcher/rhai/$f; done
ln -sfn ~/dev/nrs-org/learned-matcher/rhai/match.learned.rhai match.rhai
ln -sfn ~/dev/nrs-org/learned-matcher/target/release/libinference.so
```

Run `softmatch` from the musiclib-rs root (the bundle's default path is
relative to the CWD), or set `MUSICLIB_MATCHER_BUNDLE`. Jev refinement is on
whenever `TYPESAFE_API_KEY` is set.

## Training

```bash
cd train && uv sync
# one-time inputs: snapshot + Rhai dry run (see plan), MB facts:
#   mb_export.sql (copy to /tmp/lm/ in)  the musicbrainz docker db
D=${MUSICLIB_DATA:-../../musiclib-rs/data}
uv run python sample_gold.py --candidates $D/eval/candidates-rhai-2026-10-03.csv \
    --db $D/eval/live-2026-10-03.db --batch human-v1 --out $D/eval/gold
uv run python adjudicated_gold.py
uv run python generate.py
uv run python build_features.py          # downloads nothing; encoder in $D/learned-matcher/encoder
uv run python train.py --name v5
uv run python evaluate.py --batch adjudicated-v34 --rhai
uv run python evaluate.py --batch adjudicated-v34 --pred $D/learned-matcher/models/v5/gold_preds.csv
# after labeling: ArtifactData list → raw dir, then
uv run python collect_labels.py human-v1 <raw dir>
uv run python evaluate.py --batch human-v1 --pred $D/learned-matcher/models/v5/gold_preds.csv
```
