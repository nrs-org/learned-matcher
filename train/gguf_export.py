"""Export the title encoder as a vocabulary-less GGUF for llama.cpp (the
inference cdylib's `vulkan` encoder backend, config/inference/src/matcher/gpu.rs).

    uv run python gguf_export.py <encoder dir> [<out.gguf>]

The encoder is a BERT body with an XLM-R tokenizer, which llama.cpp's own
converter doesn't handle; the runtime tokenizes with tokenizer.json and passes
token ids, so the GGUF carries the weights only (`tokenizer.ggml.model =
none`). 2-D weights are stored f16, norms and biases f32.
"""

import json
import sys
from pathlib import Path

import gguf
import numpy as np
from safetensors.numpy import load_file


def export(encoder_dir: Path, out: Path) -> None:
    cfg = json.loads((encoder_dir / "config.json").read_text())
    if cfg.get("hidden_act") != "gelu":
        raise SystemExit(f"unsupported hidden_act {cfg.get('hidden_act')!r}")
    weights = load_file(encoder_dir / "model.safetensors")
    arch = gguf.MODEL_ARCH.BERT
    names = gguf.get_tensor_name_map(arch, cfg["num_hidden_layers"])
    w = gguf.GGUFWriter(out, gguf.MODEL_ARCH_NAMES[arch])
    w.add_name("jp-music-title-encoder")
    w.add_context_length(cfg["max_position_embeddings"])
    w.add_embedding_length(cfg["hidden_size"])
    w.add_feed_forward_length(cfg["intermediate_size"])
    w.add_block_count(cfg["num_hidden_layers"])
    w.add_head_count(cfg["num_attention_heads"])
    w.add_layer_norm_eps(cfg["layer_norm_eps"])
    w.add_causal_attention(False)
    w.add_pooling_type(gguf.PoolingType.MEAN)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    w.add_tokenizer_model("none")
    w.add_vocab_size(cfg["vocab_size"])
    w.add_token_type_count(cfg["type_vocab_size"])
    for key, t in weights.items():
        if key.startswith("pooler."):
            continue
        base, suffix = key.rsplit(".", 1)
        name = names.get_name(base)
        if name is None:
            raise SystemExit(f"no GGUF tensor name for {key}")
        f16 = t.ndim == 2 and "token_types" not in name
        w.add_tensor(f"{name}.{suffix}", t.astype(np.float16 if f16 else np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src / "model-f16.gguf"
    export(src, dst)
    print(f"wrote {dst}")
