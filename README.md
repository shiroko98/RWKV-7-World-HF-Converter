# RWKV7 Native `.pth` to vLLM-Compatible HF Converter

This repo converts the latest native RWKV7 `.pth` checkpoints into a Hugging Face style directory whose weight names match vLLM's current RWKV7 loader.

## What it does

- Reads native RWKV7 checkpoints using the latest naming style from `rwkv7-g1f-13.3b.txt`
- Maps weights to the names expected by `vllm.model_executor.models.rwkv7`
- Writes `safetensors` shards plus `model.safetensors.index.json` when needed
- Generates `config.json`, `generation_config.json`, `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`
- Copies `rwkv_vocab_v20260603.txt`, `chat_template.jinja`, and `hf_rwkv_tokenizer.py`

The converter does not depend on Triton or FLA. It only prepares a vLLM-aligned model directory.

## Default assets

- Vocabulary: `rwkv_vocab_v20260603.txt`
- Chat template: `chat_template.jinja`
- Tokenizer shim: `assets/hf_rwkv_tokenizer.py`

## Usage

```powershell
D:\anaconda\envs\model\python.exe converter.py `
  --source D:\fsdownload\rwkv7-g0b-7.2b-20251220-ctx8192.pth `
  --output-dir D:\fsdownload\rwkv7-g0b-7.2b-hf `
  --max-position-embeddings 8192 `
  --overwrite
```

Important flags:

- `--source`: native RWKV7 `.pth` or `.pt`
- `--output-dir`: target HF directory
- `--max-position-embeddings`: optional override when the checkpoint filename does not contain `ctxNNNN`
- `--max-shard-size`: shard size for safetensors output, default `5GB`
- `--overwrite`: replace an existing output directory

## vLLM example

After conversion, the output directory can be used directly as both the model path and tokenizer path:

```bash
vllm serve /mnt/d/fsdownload/rwkv7-g0b-7.2b-hf \
  --tokenizer /mnt/d/fsdownload/rwkv7-g0b-7.2b-hf
```

## Tests

```powershell
D:\anaconda\envs\model\python.exe -m pytest tests --cov=converter --cov-report=term-missing --cov-fail-under=95
```
