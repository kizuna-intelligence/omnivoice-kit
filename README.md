# omnivoice-kit

Thin training and inference kit around OmniVoice.

This repository keeps upstream OmniVoice as a git submodule under `third_party/OmniVoice`
and adds a small Python API for:

- dataset split generation
- tokenizer manifest generation
- transcript-only LoRA config generation
- LoRA inference

## Layout

```text
third_party/OmniVoice/   # upstream submodule
src/omnivoice_kit/       # thin wrapper layer
examples/                # small usage examples
```

## Setup

```bash
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate
pip install -e third_party/OmniVoice
pip install -e .
```

## Quick start

Generate transcript-only train/dev JSONL:

```bash
omnivoice-kit prepare-jsonl \
  --archive /path/to/mio_regenerated_ljspeech.tar.zst \
  --output-dir work/dataset
```

Tokenize train/dev JSONL into OmniVoice manifests:

```bash
omnivoice-kit tokenize-jsonl \
  --input-jsonl work/dataset/jsonl/train.jsonl \
  --output-dir work/tokens/train
```

Write transcript-only LoRA configs:

```bash
omnivoice-kit write-text-only-configs \
  --train-manifest work/tokens/train/data.lst \
  --dev-manifest work/tokens/dev/data.lst \
  --output-dir configs
```

Launch training:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
omnivoice-kit launch-train \
  --train-config configs/train_config_text_only_lora.json \
  --data-config configs/data_config_text_only_lora.json \
  --output-dir artifacts/train_run \
  --gpu-ids 0 \
  --num-processes 1
```

Inference with an existing LoRA checkpoint:

```bash
omnivoice-kit generate \
  --base-model k2-fsa/OmniVoice \
  --checkpoint-dir /path/to/checkpoint-300 \
  --input-jsonl examples/english_prompts.jsonl \
  --output-dir artifacts/generate_en \
  --language en
```

Smoke-tested outputs in this repo:

- training smoke checkpoint: `artifacts/smoke_train/checkpoint-1`
- inference smoke wav: `artifacts/smoke_generate_from_checkpoint1/wav/01_smoke_001.wav`

## Notes

- OmniVoice itself is kept in `third_party/OmniVoice`.
- `launch-train` uses `omnivoice_kit.train_entry` so dataloaders are prepared through `accelerate`.
- This repo does not vendor OpenVoice. Speaker similarity evaluation should be treated as optional.
