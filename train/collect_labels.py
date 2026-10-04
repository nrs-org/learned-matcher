"""Fold labeling-page documents (saved by ArtifactData list --out_dir) into
data/eval/gold/<batch>.labels.jsonl."""
import json, sys
from pathlib import Path

from paths import DATA

batch, raw = sys.argv[1], Path(sys.argv[2])
out = DATA / f"eval/gold/{batch}.labels.jsonl"
rows = []
for f in sorted(raw.rglob("*.json")):
    d = json.loads(f.read_text())
    d = d.get("data", d)
    if d.get("identity"):
        rows.append({k: d.get(k) for k in ("item_id", "type", "identity", "kind", "note", "updated_at")})
out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
print(f"{len(rows)} labels -> {out}")
