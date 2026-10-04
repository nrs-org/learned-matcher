# learned-matcher

Learned softmatch pair classifier. Plan and results: `docs/plan-learned-matcher.md`.

```bash
cd train/learned-matcher && uv sync
# one-time inputs: snapshot + Rhai dry run (see plan), MB facts:
#   mb_export.sql (copy to /tmp/lm/ in)  the musicbrainz docker db
uv run python sample_gold.py --candidates ../../data/eval/candidates-rhai-2026-10-03.csv \
    --db ../../data/eval/live-2026-10-03.db --batch human-v1 --out ../../data/eval/gold
uv run python adjudicated_gold.py
uv run python generate.py
uv run python build_features.py          # downloads nothing; encoder in data/learned-matcher/encoder
uv run python train.py --name v5
uv run python evaluate.py --batch adjudicated-v34 --rhai
uv run python evaluate.py --batch adjudicated-v34 --pred ../../data/learned-matcher/models/v5/gold_preds.csv
# after labeling: ArtifactData list → raw dir, then
uv run python collect_labels.py human-v1 <raw dir>
uv run python evaluate.py --batch human-v1 --pred ../../data/learned-matcher/models/v5/gold_preds.csv
```
