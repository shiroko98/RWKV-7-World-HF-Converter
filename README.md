# RWKV7 Native `.pth` to vLLM/Transformers-Compatible HF Converter

This repo converts the latest native RWKV7 `.pth` checkpoints into a Hugging Face style directory whose weight names match vLLM's current RWKV7 loader. The converted directory also includes Transformers remote-code files for pure PyTorch loading with `trust_remote_code=True`.

## What it does

- Reads native RWKV7 checkpoints using the latest naming style from `rwkv7-g1f-13.3b.txt`
- Maps weights to the names expected by `vllm.model_executor.models.rwkv7`
- Writes `safetensors` shards plus `model.safetensors.index.json` when needed
- Generates `config.json`, `generation_config.json`, `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`
- Copies `rwkv_vocab_v20260603.txt`, `chat_template.jinja`, `hf_rwkv_tokenizer.py`, `configuration_rwkv7.py`, and `modeling_rwkv7.py`

The converter and the Transformers model shim do not depend on Triton or FLA. The safetensors weight names remain vLLM-aligned, and the Transformers implementation is adapted to that naming.

## Default assets

- Vocabulary: `rwkv_vocab_v20260603.txt`
- Chat template: `chat_template.jinja`
- Tokenizer shim: `assets/hf_rwkv_tokenizer.py`
- Transformers config shim: `assets/configuration_rwkv7.py`
- Transformers model shim: `assets/modeling_rwkv7.py`

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
- `--max-position-embeddings`: optional override when the checkpoint filename does not contain `ctxNNNN`; otherwise defaults to `86016`
- `--max-shard-size`: shard size for safetensors output, default `5GB`
- `--overwrite`: replace an existing output directory

## vLLM example

After conversion, the output directory can be used directly as both the model path and tokenizer path:

```bash
vllm serve /mnt/d/fsdownload/rwkv7-g0b-7.2b-hf \
  --tokenizer /mnt/d/fsdownload/rwkv7-g0b-7.2b-hf
```

## Transformers example

The output directory can also be loaded by Transformers with remote code:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "/mnt/d/fsdownload/rwkv7-g0b-7.2b-hf"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

inputs = tokenizer("Hello RWKV", return_tensors="pt")
outputs = model(**inputs)
```

The Transformers implementation is a plain PyTorch recurrent path intended for compatibility, validation, and simple inference. For high-throughput serving, use the vLLM RWKV7 implementation.

## Tests

```powershell
D:\anaconda\envs\model\python.exe -m pytest tests --cov=converter --cov-report=term-missing --cov-fail-under=95
```
