"""Assemble the runtime bundle the inference cdylib loads (`inference_matcher_open`).

    uv run python bundle.py [--model v16 --rel rel-v7] [--out DIR]

Default output: data/learned-matcher/bundle/<model>/
  manifest.json                     names, thresholds, kinds, output layout
  model.txt                         main 4-class LightGBM model
  kind.txt direction.txt structure.txt kinds.json   relation heads
  encoder/{config.json,tokenizer.json,model.safetensors}
  encoder/model-f16.gguf            the same encoder for llama.cpp (GPU backend)
"""

import argparse
import json
import shutil

from features import ROOT
from gguf_export import export as export_gguf

OUT = ROOT / "data/learned-matcher"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v16")
    ap.add_argument("--rel", default="rel-v7")
    ap.add_argument("--out")
    args = ap.parse_args()
    mdir, rdir = OUT / "models" / args.model, OUT / "models" / args.rel
    out = OUT / "bundle" / args.model if args.out is None else ROOT / args.out
    (out / "encoder").mkdir(parents=True, exist_ok=True)
    shutil.copy(mdir / "model.txt", out / "model.txt")
    for f in ("kind.txt", "direction.txt", "structure.txt", "kinds.json"):
        shutil.copy(rdir / f, out / f)
    for f in ("config.json", "tokenizer.json", "model.safetensors"):
        shutil.copy(OUT / "encoder" / f, out / "encoder" / f)
    export_gguf(out / "encoder", out / "encoder" / "model-f16.gguf")
    kinds = json.loads((rdir / "kinds.json").read_text())
    manifest = {
        "schema": "musiclib-learned-matcher-bundle/1",
        "model": args.model,
        "rel": args.rel,
        "facts_schema": "musiclib-pair-facts/1",
        "encoder": "gbnam8/jp-music-title-encoder (v7-L6), 256-dim, truncation 48",
        "classes": ["same", "related", "sibling", "unrelated"],
        "kinds": kinds,
        "thresholds": json.loads((mdir / "thresholds.json").read_text()),
        "outputs": ["p_same", "p_related", "p_sibling", "p_unrelated", "guard",
                    "p_sibling_structure", "p_a_derived"] + [f"p_kind_{k}" for k in kinds],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
