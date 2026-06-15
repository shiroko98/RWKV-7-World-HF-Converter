from __future__ import annotations

import argparse
from ast import literal_eval
import json
import re
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import save_file


DEFAULT_ROOT = Path(__file__).resolve().parent
DEFAULT_VOCAB_SOURCE = DEFAULT_ROOT / "rwkv_vocab_v20260603.txt"
DEFAULT_CHAT_TEMPLATE_SOURCE = DEFAULT_ROOT / "chat_template.jinja"
DEFAULT_TOKENIZER_SOURCE = DEFAULT_ROOT / "assets" / "hf_rwkv_tokenizer.py"
CANONICAL_VOCAB_NAME = "rwkv_vocab_v20260603.txt"
MODEL_INDEX_NAME = "model.safetensors.index.json"
BLOCK_RE = re.compile(r"blocks\.(\d+)\.(.+)")
CTX_RE = re.compile(r"ctx(\d+)")

TOP_LEVEL_NAME_MAP = {
    "emb.weight": "model.embed_tokens.weight",
    "ln_out.weight": "model.norm.weight",
    "ln_out.bias": "model.norm.bias",
    "head.weight": "lm_head.weight",
}

BLOCK_NAME_MAP = {
    "ln1.weight": "attn_norm.weight",
    "ln1.bias": "attn_norm.bias",
    "ln2.weight": "ffn_norm.weight",
    "ln2.bias": "ffn_norm.bias",
    "att.x_r": "attn.x_r",
    "att.x_w": "attn.x_w",
    "att.x_k": "attn.x_k",
    "att.x_v": "attn.x_v",
    "att.x_a": "attn.x_a",
    "att.x_g": "attn.x_g",
    "att.k_k": "attn.k_k",
    "att.k_a": "attn.k_a",
    "att.r_k": "attn.r_k",
    "att.receptance.weight": "attn.r_proj.weight",
    "att.key.weight": "attn.k_proj.weight",
    "att.value.weight": "attn.v_proj.weight",
    "att.output.weight": "attn.o_proj.weight",
    "att.ln_x.weight": "attn.g_norm.weight",
    "att.ln_x.bias": "attn.g_norm.bias",
    "ffn.x_k": "ffn.x_k",
    "ffn.key.weight": "ffn.key.weight",
    "ffn.value.weight": "ffn.value.weight",
}

LORA_SPECS = {
    "att.w1": ("attn.w_lora.lora.0.weight", "transpose"),
    "att.w2": ("attn.w_lora.lora.2.weight", "transpose"),
    "att.w0": ("attn.w_lora.lora.2.bias", "squeeze"),
    "att.a1": ("attn.a_lora.lora.0.weight", "transpose"),
    "att.a2": ("attn.a_lora.lora.2.weight", "transpose"),
    "att.a0": ("attn.a_lora.lora.2.bias", "squeeze"),
    "att.g1": ("attn.g_lora.lora.0.weight", "transpose"),
    "att.g2": ("attn.g_lora.lora.2.weight", "transpose"),
    "att.v1": ("attn.v_lora.lora.0.weight", "transpose"),
    "att.v2": ("attn.v_lora.lora.2.weight", "transpose"),
    "att.v0": ("attn.v_lora.lora.2.bias", "squeeze"),
}

SPECIAL_TOKEN_FALLBACKS = (
    "<|endoftext|>",
    "<|rwkv_tokenizer_end_of_text|>",
    "<s>",
)

EXTRA_SPECIAL_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<|think|>",
    "<|tool_call|>",
]


class ConversionError(ValueError):
    """Raised when the native checkpoint cannot be converted safely."""


@dataclass(frozen=True)
class ConversionSummary:
    source: Path
    output_dir: Path
    num_layers: int
    num_tensors: int
    num_shards: int


def load_torch_file(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility for older torch versions
        return torch.load(path, map_location="cpu")


def looks_like_native_rwkv7_state_dict(state_dict: dict[str, Any]) -> bool:
    required_keys = {"emb.weight", "ln_out.weight", "head.weight"}
    if not required_keys.issubset(state_dict):
        return False
    return (
        "blocks.0.att.receptance.weight" in state_dict
        and "blocks.0.ffn.key.weight" in state_dict
    )


def extract_native_rwkv7_state_dict(loaded: Any) -> dict[str, torch.Tensor] | None:
    if isinstance(loaded, dict):
        if looks_like_native_rwkv7_state_dict(loaded):
            return loaded
        for key in ("state_dict", "model", "weights"):
            nested = loaded.get(key)
            if isinstance(nested, dict) and looks_like_native_rwkv7_state_dict(nested):
                return nested
    return None


def load_native_rwkv7_state_dict(source: Path) -> dict[str, torch.Tensor]:
    loaded = load_torch_file(source)
    state_dict = extract_native_rwkv7_state_dict(loaded)
    if state_dict is None:
        raise ConversionError(
            f"{source} does not look like a native RWKV7 checkpoint with latest naming."
        )
    return state_dict


def transform_tensor(tensor: torch.Tensor, transform: str | None) -> torch.Tensor:
    if transform == "transpose":
        return tensor.transpose(0, 1)
    if transform == "squeeze":
        return tensor.reshape(-1)
    return tensor


def map_native_weight(name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    if name in TOP_LEVEL_NAME_MAP:
        return [(TOP_LEVEL_NAME_MAP[name], tensor)]

    match = BLOCK_RE.fullmatch(name)
    if match is None:
        raise ConversionError(f"Unsupported native RWKV7 parameter: {name}")

    layer_idx = int(match.group(1))
    suffix = match.group(2)
    prefix = f"model.layers.{layer_idx}"

    if suffix == "ln0.weight" and layer_idx == 0:
        return [(f"{prefix}.pre_norm.weight", tensor)]
    if suffix == "ln0.bias" and layer_idx == 0:
        return [(f"{prefix}.pre_norm.bias", tensor)]

    mapped_name = BLOCK_NAME_MAP.get(suffix)
    if mapped_name is not None:
        transform = "squeeze" if suffix in {"att.k_k", "att.k_a", "ffn.x_k"} else None
        return [(f"{prefix}.{mapped_name}", transform_tensor(tensor, transform))]

    lora_spec = LORA_SPECS.get(suffix)
    if lora_spec is None:
        raise ConversionError(f"Unsupported RWKV7 block parameter: {name}")

    if suffix.startswith("att.v") and layer_idx == 0:
        return []

    mapped_name, transform = lora_spec
    return [(f"{prefix}.{mapped_name}", transform_tensor(tensor, transform))]


def map_native_state_dict(
    state_dict: dict[str, torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    converted: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, tensor in state_dict.items():
        for mapped_name, mapped_tensor in map_native_weight(name, tensor):
            converted[mapped_name] = mapped_tensor
    return converted


def iter_layer_indices(state_dict: dict[str, torch.Tensor]) -> list[int]:
    layer_indices = sorted(
        {
            int(match.group(1))
            for key in state_dict
            if (match := BLOCK_RE.match(key)) is not None
        }
    )
    if not layer_indices:
        raise ConversionError("Unable to infer layer count from checkpoint.")
    return layer_indices


def infer_max_position_embeddings(
    source: Path, override: int | None = None
) -> int:
    if override is not None:
        return override
    match = CTX_RE.search(source.stem)
    if match is not None:
        return int(match.group(1))
    return 2048


def torch_dtype_to_hf_string(dtype: torch.dtype) -> str:
    if dtype is torch.float16:
        return "float16"
    if dtype is torch.float32:
        return "float32"
    if dtype is torch.bfloat16:
        return "bfloat16"
    return str(dtype).removeprefix("torch.")


def read_vocab_token_ids(vocab_path: Path) -> dict[str, int]:
    token_ids: dict[str, int] = {}
    with vocab_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            token_id = int(line[: line.index(" ")])
            token = literal_eval(line[line.index(" ") : line.rindex(" ")])
            if isinstance(token, str):
                token_text = token
            else:
                try:
                    token_text = token.decode("utf-8")
                except UnicodeDecodeError:
                    continue
            if token_text in SPECIAL_TOKEN_FALLBACKS or token_text in EXTRA_SPECIAL_TOKENS:
                token_ids[token_text] = token_id
    return token_ids


def resolve_primary_special_token(token_ids: dict[str, int]) -> str:
    for token in SPECIAL_TOKEN_FALLBACKS:
        if token in token_ids:
            return token
    raise ConversionError(
        "Vocabulary is missing a supported end-of-text token "
        f"from {SPECIAL_TOKEN_FALLBACKS!r}."
    )


def build_tokenizer_files(
    vocab_path: Path,
    chat_template_path: Path,
) -> tuple[dict[str, int], dict[str, Any], dict[str, Any], dict[str, int]]:
    token_ids = read_vocab_token_ids(vocab_path)
    primary_special_token = resolve_primary_special_token(token_ids)
    available_extra_tokens = [token for token in EXTRA_SPECIAL_TOKENS if token in token_ids]
    chat_template = chat_template_path.read_text(encoding="utf-8")

    tokenizer_config = {
        "add_prefix_space": False,
        "tokenizer_class": "RwkvTokenizer",
        "use_fast": False,
        "clean_up_tokenization_spaces": False,
        "auto_map": {
            "AutoTokenizer": [
                "hf_rwkv_tokenizer.RwkvTokenizer",
                None,
            ]
        },
        "bos_token": primary_special_token,
        "eos_token": primary_special_token,
        "unk_token": primary_special_token,
        "pad_token": primary_special_token,
        "additional_special_tokens": available_extra_tokens,
        "model_max_length": 1_000_000_000_000,
        "chat_template": chat_template,
        "added_tokens_decoder": {
            str(token_id): {
                "content": token,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            }
            for token, token_id in token_ids.items()
        },
    }

    special_tokens_map = {
        "bos_token": primary_special_token,
        "eos_token": primary_special_token,
        "unk_token": primary_special_token,
        "pad_token": primary_special_token,
        "additional_special_tokens": available_extra_tokens,
    }

    explicit_ids = {
        "bos_token_id": token_ids[primary_special_token],
        "eos_token_id": token_ids[primary_special_token],
        "pad_token_id": token_ids[primary_special_token],
    }

    return token_ids, tokenizer_config, special_tokens_map, explicit_ids


def build_model_config(
    state_dict: dict[str, torch.Tensor],
    source: Path,
    special_token_ids: dict[str, int],
    max_position_embeddings: int | None = None,
) -> dict[str, Any]:
    layer_indices = iter_layer_indices(state_dict)
    hidden_size = int(state_dict["emb.weight"].shape[1])
    num_hidden_layers = layer_indices[-1] + 1
    r_k = state_dict["blocks.0.att.r_k"]
    intermediate_size = int(state_dict["blocks.0.ffn.key.weight"].shape[0])
    hidden_ratio = (
        intermediate_size // hidden_size
        if intermediate_size % hidden_size == 0
        else None
    )
    v_lora_rank = 0
    for layer_idx in layer_indices:
        key = f"blocks.{layer_idx}.att.v1"
        if key in state_dict:
            v_lora_rank = int(state_dict[key].shape[1])
            break

    return {
        "model_type": "rwkv7",
        "architectures": ["RWKV7ForCausalLM"],
        "attn_mode": "chunk",
        "hidden_size": hidden_size,
        "hidden_ratio": hidden_ratio,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_hidden_layers,
        "head_dim": int(r_k.shape[1]),
        "num_heads": int(r_k.shape[0]),
        "value_dim": [
            int(state_dict[f"blocks.{layer_idx}.att.output.weight"].shape[1])
            for layer_idx in layer_indices
        ],
        "decay_low_rank_dim": int(state_dict["blocks.0.att.w1"].shape[1]),
        "gate_low_rank_dim": int(state_dict["blocks.0.att.g1"].shape[1]),
        "a_low_rank_dim": int(state_dict["blocks.0.att.a1"].shape[1]),
        "v_low_rank_dim": v_lora_rank,
        "hidden_act": "sqrelu",
        "max_position_embeddings": infer_max_position_embeddings(
            source, max_position_embeddings
        ),
        "norm_first": True,
        "norm_bias": "ln_out.bias" in state_dict,
        "norm_eps": 1e-5,
        "tie_word_embeddings": False,
        "use_cache": True,
        "initializer_range": 0.02,
        "fuse_norm": False,
        "vocab_size": int(state_dict["emb.weight"].shape[0]),
        "torch_dtype": torch_dtype_to_hf_string(state_dict["emb.weight"].dtype),
        **special_token_ids,
    }


def build_generation_config(
    special_token_ids: dict[str, int], max_position_embeddings: int
) -> dict[str, Any]:
    return {
        "chat_format": "chatml",
        "eos_token_id": special_token_ids["eos_token_id"],
        "pad_token_id": special_token_ids["pad_token_id"],
        "max_window_size": max_position_embeddings,
        "max_new_tokens": max_position_embeddings,
        "do_sample": True,
        "top_k": 0,
        "top_p": 0.1,
        "repetition_penalty": 1.0,
    }


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ConversionError(f"Output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            if not overwrite:
                raise ConversionError(
                    f"Output directory already exists and is not empty: {output_dir}"
                )
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_weight_files(
    state_dict: OrderedDict[str, torch.Tensor],
    output_dir: Path,
    max_shard_size: str,
) -> int:
    state_dict_split = split_torch_state_dict_into_shards(
        state_dict,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size=max_shard_size,
    )

    for filename, tensor_names in state_dict_split.filename_to_tensors.items():
        shard = {
            tensor_name: state_dict[tensor_name].contiguous()
            for tensor_name in tensor_names
        }
        save_file(shard, str(output_dir / filename), metadata={"format": "pt"})

    if state_dict_split.is_sharded:
        write_json(
            output_dir / MODEL_INDEX_NAME,
            {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            },
        )

    return len(state_dict_split.filename_to_tensors)


def convert_checkpoint(
    source: Path,
    output_dir: Path,
    *,
    vocab_path: Path = DEFAULT_VOCAB_SOURCE,
    chat_template_path: Path = DEFAULT_CHAT_TEMPLATE_SOURCE,
    tokenizer_source: Path = DEFAULT_TOKENIZER_SOURCE,
    max_shard_size: str = "5GB",
    max_position_embeddings: int | None = None,
    overwrite: bool = False,
) -> ConversionSummary:
    source = source.resolve()
    output_dir = output_dir.resolve()
    prepare_output_dir(output_dir, overwrite)

    state_dict = load_native_rwkv7_state_dict(source)
    token_ids, tokenizer_config, special_tokens_map, explicit_special_ids = (
        build_tokenizer_files(vocab_path, chat_template_path)
    )
    model_config = build_model_config(
        state_dict,
        source,
        explicit_special_ids,
        max_position_embeddings=max_position_embeddings,
    )
    generation_config = build_generation_config(
        explicit_special_ids,
        model_config["max_position_embeddings"],
    )
    converted_state_dict = map_native_state_dict(state_dict)
    num_shards = write_weight_files(converted_state_dict, output_dir, max_shard_size)

    shutil.copyfile(vocab_path, output_dir / CANONICAL_VOCAB_NAME)
    shutil.copyfile(chat_template_path, output_dir / "chat_template.jinja")
    shutil.copyfile(tokenizer_source, output_dir / "hf_rwkv_tokenizer.py")
    (output_dir / "__init__.py").write_text("", encoding="utf-8")
    write_json(output_dir / "config.json", model_config)
    write_json(output_dir / "generation_config.json", generation_config)
    write_json(output_dir / "tokenizer_config.json", tokenizer_config)
    write_json(output_dir / "special_tokens_map.json", special_tokens_map)
    write_json(output_dir / "added_tokens.json", {})

    return ConversionSummary(
        source=source,
        output_dir=output_dir,
        num_layers=model_config["num_hidden_layers"],
        num_tensors=len(converted_state_dict),
        num_shards=num_shards,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert native RWKV7 .pth checkpoints into vLLM-compatible HF directories."
    )
    parser.add_argument("--source", required=True, help="Path to native RWKV7 .pth or .pt checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Output HF directory.")
    parser.add_argument(
        "--vocab-file",
        default=str(DEFAULT_VOCAB_SOURCE),
        help="Vocabulary txt to copy into the output directory.",
    )
    parser.add_argument(
        "--chat-template",
        default=str(DEFAULT_CHAT_TEMPLATE_SOURCE),
        help="Jinja chat template to embed into tokenizer_config.json.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Maximum safetensors shard size, for example 2GB or 500MB.",
    )
    parser.add_argument(
        "--max-position-embeddings",
        type=int,
        default=None,
        help="Optional override for context length when the filename does not contain ctxNNNN.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing non-empty output directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = convert_checkpoint(
        Path(args.source),
        Path(args.output_dir),
        vocab_path=Path(args.vocab_file),
        chat_template_path=Path(args.chat_template),
        max_shard_size=args.max_shard_size,
        max_position_embeddings=args.max_position_embeddings,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "source": str(summary.source),
                "output_dir": str(summary.output_dir),
                "num_layers": summary.num_layers,
                "num_tensors": summary.num_tensors,
                "num_shards": summary.num_shards,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
