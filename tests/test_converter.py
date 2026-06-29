from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

import converter


def build_native_state_dict(
    *,
    num_layers: int = 2,
    hidden_size: int = 8,
    vocab_size: int = 64,
    head_dim: int = 4,
    ffn_multiplier: int = 2,
    w_rank: int = 2,
    a_rank: int = 3,
    g_rank: int = 4,
    v_rank: int = 5,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {
        "emb.weight": torch.arange(vocab_size * hidden_size, dtype=torch.float32).reshape(vocab_size, hidden_size).to(dtype),
        "ln_out.weight": torch.ones(hidden_size, dtype=dtype),
        "ln_out.bias": torch.zeros(hidden_size, dtype=dtype),
        "head.weight": torch.ones(vocab_size, hidden_size, dtype=dtype),
    }
    for layer_idx in range(num_layers):
        prefix = f"blocks.{layer_idx}"
        if layer_idx == 0:
            state_dict[f"{prefix}.ln0.weight"] = torch.full((hidden_size,), 1.0, dtype=dtype)
            state_dict[f"{prefix}.ln0.bias"] = torch.full((hidden_size,), 2.0, dtype=dtype)
        state_dict[f"{prefix}.ln1.weight"] = torch.full((hidden_size,), 3.0 + layer_idx, dtype=dtype)
        state_dict[f"{prefix}.ln1.bias"] = torch.full((hidden_size,), 4.0 + layer_idx, dtype=dtype)
        state_dict[f"{prefix}.ln2.weight"] = torch.full((hidden_size,), 5.0 + layer_idx, dtype=dtype)
        state_dict[f"{prefix}.ln2.bias"] = torch.full((hidden_size,), 6.0 + layer_idx, dtype=dtype)

        for name, base in (
            ("x_r", 0),
            ("x_w", 1),
            ("x_k", 2),
            ("x_v", 3),
            ("x_a", 4),
            ("x_g", 5),
        ):
            state_dict[f"{prefix}.att.{name}"] = torch.full((1, 1, hidden_size), base + layer_idx, dtype=dtype)

        state_dict[f"{prefix}.att.w0"] = torch.arange(hidden_size, dtype=torch.float32).reshape(1, 1, hidden_size).to(dtype)
        state_dict[f"{prefix}.att.w1"] = torch.arange(hidden_size * w_rank, dtype=torch.float32).reshape(hidden_size, w_rank).to(dtype)
        state_dict[f"{prefix}.att.w2"] = torch.arange(w_rank * hidden_size, dtype=torch.float32).reshape(w_rank, hidden_size).to(dtype)
        state_dict[f"{prefix}.att.a0"] = torch.arange(hidden_size, dtype=torch.float32).reshape(1, 1, hidden_size).to(dtype)
        state_dict[f"{prefix}.att.a1"] = torch.arange(hidden_size * a_rank, dtype=torch.float32).reshape(hidden_size, a_rank).to(dtype)
        state_dict[f"{prefix}.att.a2"] = torch.arange(a_rank * hidden_size, dtype=torch.float32).reshape(a_rank, hidden_size).to(dtype)
        state_dict[f"{prefix}.att.g1"] = torch.arange(hidden_size * g_rank, dtype=torch.float32).reshape(hidden_size, g_rank).to(dtype)
        state_dict[f"{prefix}.att.g2"] = torch.arange(g_rank * hidden_size, dtype=torch.float32).reshape(g_rank, hidden_size).to(dtype)
        state_dict[f"{prefix}.att.v0"] = torch.arange(hidden_size, dtype=torch.float32).reshape(1, 1, hidden_size).to(dtype)
        state_dict[f"{prefix}.att.v1"] = torch.arange(hidden_size * v_rank, dtype=torch.float32).reshape(hidden_size, v_rank).to(dtype)
        state_dict[f"{prefix}.att.v2"] = torch.arange(v_rank * hidden_size, dtype=torch.float32).reshape(v_rank, hidden_size).to(dtype)
        state_dict[f"{prefix}.att.k_k"] = torch.arange(hidden_size, dtype=torch.float32).reshape(1, 1, hidden_size).to(dtype)
        state_dict[f"{prefix}.att.k_a"] = torch.arange(hidden_size, dtype=torch.float32).reshape(1, 1, hidden_size).to(dtype)
        state_dict[f"{prefix}.att.r_k"] = torch.arange((hidden_size // head_dim) * head_dim, dtype=torch.float32).reshape(hidden_size // head_dim, head_dim).to(dtype)
        state_dict[f"{prefix}.att.receptance.weight"] = torch.ones(hidden_size, hidden_size, dtype=dtype)
        state_dict[f"{prefix}.att.key.weight"] = torch.ones(hidden_size, hidden_size, dtype=dtype) * 2
        state_dict[f"{prefix}.att.value.weight"] = torch.ones(hidden_size, hidden_size, dtype=dtype) * 3
        state_dict[f"{prefix}.att.output.weight"] = torch.ones(hidden_size, hidden_size, dtype=dtype) * 4
        state_dict[f"{prefix}.att.ln_x.weight"] = torch.ones(hidden_size, dtype=dtype) * 5
        state_dict[f"{prefix}.att.ln_x.bias"] = torch.ones(hidden_size, dtype=dtype) * 6
        state_dict[f"{prefix}.ffn.x_k"] = torch.arange(hidden_size, dtype=torch.float32).reshape(1, 1, hidden_size).to(dtype)
        state_dict[f"{prefix}.ffn.key.weight"] = torch.ones(hidden_size * ffn_multiplier, hidden_size, dtype=dtype)
        state_dict[f"{prefix}.ffn.value.weight"] = torch.ones(hidden_size, hidden_size * ffn_multiplier, dtype=dtype)
    return state_dict


def write_vocab(path: Path) -> None:
    lines = [
        "1 'a' 1",
        "2 'b' 1",
        "3 '\\n\\n' 2",
        "65530 '<|im_start|>' 12",
        "65531 '<|im_end|>' 10",
        "65532 '<|endoftext|>' 13",
        "65533 '<think>' 7",
        "65534 '<tool_call>' 11",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_looks_like_native_rwkv7_state_dict_requires_core_keys():
    assert not converter.looks_like_native_rwkv7_state_dict({"emb.weight": torch.ones(1)})
    assert converter.looks_like_native_rwkv7_state_dict(build_native_state_dict())


def test_extract_native_rwkv7_state_dict_handles_wrapped_checkpoints():
    state_dict = build_native_state_dict()
    assert converter.extract_native_rwkv7_state_dict({"state_dict": state_dict}) == state_dict
    assert converter.extract_native_rwkv7_state_dict({"model": state_dict}) == state_dict
    assert converter.extract_native_rwkv7_state_dict({"weights": state_dict}) == state_dict
    assert converter.extract_native_rwkv7_state_dict(state_dict) == state_dict
    assert converter.extract_native_rwkv7_state_dict({"unexpected": state_dict}) is None


def test_map_native_state_dict_matches_vllm_names_and_skips_layer0_v_lora():
    converted = converter.map_native_state_dict(build_native_state_dict())

    assert "model.embed_tokens.weight" in converted
    assert "model.norm.weight" in converted
    assert "model.layers.0.pre_norm.weight" in converted
    assert "model.layers.0.attn.x_r" in converted
    assert "model.layers.0.attn.k_k" in converted
    assert converted["model.layers.0.attn.k_k"].shape == (8,)
    assert converted["model.layers.1.attn.w_lora.lora.0.weight"].shape == (2, 8)
    assert converted["model.layers.1.attn.v_lora.lora.0.weight"].shape == (5, 8)
    assert "model.layers.0.attn.v_lora.lora.0.weight" not in converted
    assert "model.layers.0.attn.x_x" not in converted


def test_map_native_weight_rejects_unknown_latest_name():
    with pytest.raises(converter.ConversionError):
        converter.map_native_weight("bad.top.level", torch.ones(1))
    with pytest.raises(converter.ConversionError):
        converter.map_native_weight("blocks.0.att.not_real", torch.ones(1))


def test_infer_max_position_embeddings_prefers_override_and_ctx_hint():
    path = Path("rwkv7-sample-ctx8192.pth")
    assert converter.infer_max_position_embeddings(path, None) == 8192
    assert converter.infer_max_position_embeddings(path, 32768) == 32768
    assert converter.infer_max_position_embeddings(Path("rwkv7-sample.pth"), None) == 86016


def test_build_tokenizer_files_requires_supported_eot_token(tmp_path: Path):
    vocab_path = tmp_path / "bad_vocab.txt"
    vocab_path.write_text("1 'a' 1\n", encoding="utf-8")
    chat_template_path = tmp_path / "chat_template.jinja"
    chat_template_path.write_text("{{ messages }}", encoding="utf-8")

    with pytest.raises(converter.ConversionError):
        converter.build_tokenizer_files(vocab_path, chat_template_path)


def test_read_vocab_token_ids_skips_binary_tokens_and_prefers_real_special_tokens(tmp_path: Path):
    vocab_path = tmp_path / "rwkv_vocab_v20260603.txt"
    vocab_path.write_text(
        "\n".join(
            [
                "1 b'\\x80' 1",
                "2 '<|im_start|>' 12",
                "3 '<|endoftext|>' 13",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    token_ids = converter.read_vocab_token_ids(vocab_path)

    assert token_ids["<|im_start|>"] == 2
    assert token_ids["<|endoftext|>"] == 3
    assert "\x80" not in token_ids


def test_prepare_output_dir_requires_overwrite_for_non_empty_dir(tmp_path: Path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "marker.txt").write_text("x", encoding="utf-8")

    with pytest.raises(converter.ConversionError):
        converter.prepare_output_dir(output_dir, overwrite=False)

    converter.prepare_output_dir(output_dir, overwrite=True)
    assert output_dir.exists()
    assert list(output_dir.iterdir()) == []


def test_convert_checkpoint_writes_vllm_ready_hf_directory(tmp_path: Path, capsys):
    checkpoint_path = tmp_path / "rwkv7-demo-ctx4096.pth"
    torch.save({"state_dict": build_native_state_dict()}, checkpoint_path)

    vocab_path = tmp_path / "rwkv_vocab_v20260603.txt"
    write_vocab(vocab_path)
    chat_template_path = tmp_path / "chat_template.jinja"
    chat_template_path.write_text("{{ '<|im_start|>User: ' + messages[0]['content'] }}", encoding="utf-8")

    output_dir = tmp_path / "converted"
    exit_code = converter.main(
        [
            "--source",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--vocab-file",
            str(vocab_path),
            "--chat-template",
            str(chat_template_path),
            "--max-shard-size",
            "1KB",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["num_layers"] == 2
    assert summary["num_shards"] >= 2

    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    tokenizer_config = json.loads((output_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    index = json.loads((output_dir / converter.MODEL_INDEX_NAME).read_text(encoding="utf-8"))

    assert config["model_type"] == "rwkv7"
    assert config["architectures"] == ["RWKV7ForCausalLM"]
    assert config["max_position_embeddings"] == 4096
    assert config["bos_token_id"] == 65532
    assert config["torch_dtype"] == "bfloat16"
    assert tokenizer_config["tokenizer_class"] == "RwkvTokenizer"
    assert tokenizer_config["auto_map"]["AutoTokenizer"][0] == "hf_rwkv_tokenizer.RwkvTokenizer"
    assert tokenizer_config["chat_template"] == "{{ '<|im_start|>User: ' + messages[0]['content'] }}"
    assert "<|im_start|>" in tokenizer_config["additional_special_tokens"]
    assert "model.layers.1.attn.v_lora.lora.0.weight" in index["weight_map"]
    assert "model.layers.0.attn.v_lora.lora.0.weight" not in index["weight_map"]

    shard_names = sorted({Path(name).name for name in index["weight_map"].values()})
    shard = load_file(str(output_dir / shard_names[0]))
    all_keys = set()
    for shard_name in shard_names:
        all_keys.update(load_file(str(output_dir / shard_name)).keys())

    assert "model.embed_tokens.weight" in all_keys
    assert "model.layers.0.attn.v_lora.lora.0.weight" not in all_keys
    assert (output_dir / "rwkv_vocab_v20260603.txt").exists()
    assert (output_dir / "hf_rwkv_tokenizer.py").exists()
    assert (output_dir / "chat_template.jinja").exists()


def test_load_native_rwkv7_state_dict_rejects_non_native_checkpoint(tmp_path: Path):
    checkpoint_path = tmp_path / "not-rwkv7.pth"
    torch.save({"weight": torch.ones(1)}, checkpoint_path)

    with pytest.raises(converter.ConversionError):
        converter.load_native_rwkv7_state_dict(checkpoint_path)


def test_load_torch_file_fallback_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    checkpoint_path = tmp_path / "tiny.pth"
    torch.save({"state_dict": build_native_state_dict()}, checkpoint_path)
    original_torch_load = torch.load
    state = {"calls": 0}

    def fake_torch_load(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise TypeError("weights_only unsupported")
        return original_torch_load(*args, **kwargs)

    monkeypatch.setattr(converter.torch, "load", fake_torch_load)

    loaded = converter.load_torch_file(checkpoint_path)

    assert "state_dict" in loaded
    assert state["calls"] == 2


def test_latest_rwkv7_g1f_keyset_is_supported_by_mapper():
    sample_path = Path(__file__).resolve().parents[1] / "rwkv7-g1f-13.3b.txt"
    unsupported: list[str] = []
    for line in sample_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("Parameter: "):
            continue
        name = line.split(", Shape:", 1)[0].removeprefix("Parameter: ").strip()
        if name in converter.TOP_LEVEL_NAME_MAP:
            continue
        if name.endswith(".weight"):
            tensor = torch.ones(2, 2)
        else:
            tensor = torch.ones(1, 1, 2)
        try:
            converter.map_native_weight(name, tensor)
        except converter.ConversionError:
            unsupported.append(name)

    assert unsupported == []
