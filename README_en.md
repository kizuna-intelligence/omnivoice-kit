# omnivoice-kit

Minimal training and inference kit for OmniVoice.

- 日本語 README: [README.md](./README.md)

This repository intentionally focuses on only three tasks:

- LoRA training
- full finetune training
- inference with trained checkpoints or adapters

## Layout

```text
third_party/OmniVoice/   upstream OmniVoice
src/omnivoice_kit/       thin wrappers for training and inference
examples/                minimal input examples
```

## Setup

```bash
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate
pip install -e third_party/OmniVoice
pip install -e .
```

## Data Preparation

LJSpeech-style archive:

```bash
omnivoice-kit prepare-jsonl \
  --archive /path/to/dataset.tar.zst \
  --output-dir work/dataset
```

`VOICEACTRESS100_###/*.flac + JSON` format:

```bash
omnivoice-kit prepare-voiceactress-jsonl \
  --dataset-dir /path/to/tsukuyomichan_processed \
  --output-dir work/tsukuyomi_voiceactress100
```

Token manifest generation:

```bash
omnivoice-kit tokenize-jsonl \
  --input-jsonl work/tsukuyomi_voiceactress100/jsonl/train.jsonl \
  --output-dir work/tokens/train

omnivoice-kit tokenize-jsonl \
  --input-jsonl work/tsukuyomi_voiceactress100/jsonl/dev.jsonl \
  --output-dir work/tokens/dev
```

## LoRA Training

Generate config files:

```bash
omnivoice-kit write-lora-configs \
  --train-manifest work/tokens/train/data.lst \
  --dev-manifest work/tokens/dev/data.lst \
  --output-dir configs/lora
```

Run training:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
omnivoice-kit launch-train \
  --train-config configs/lora/train_config_lora.json \
  --data-config configs/lora/data_config_lora.json \
  --output-dir artifacts/train_lora \
  --num-processes 1
```

## Full Finetune Training

Generate config files:

```bash
omnivoice-kit write-full-finetune-configs \
  --train-manifest work/tokens/train/data.lst \
  --dev-manifest work/tokens/dev/data.lst \
  --output-dir configs/full_finetune
```

Run training:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
omnivoice-kit launch-train \
  --train-config configs/full_finetune/train_config_full_finetune.json \
  --data-config configs/full_finetune/data_config_full_finetune.json \
  --output-dir artifacts/train_full_finetune \
  --num-processes 1
```

## Inference

The same `generate` command can load either a LoRA adapter or a full checkpoint.

```bash
omnivoice-kit generate \
  --base-model k2-fsa/OmniVoice \
  --checkpoint-dir /path/to/checkpoint-or-adapter \
  --input-jsonl examples/japanese_prompts.jsonl \
  --output-dir artifacts/generate_ja \
  --language ja \
  --num-step 16
```

## Current Recommendation For Tsukuyomichan TTS

The current top recommendation is the full finetune setting `full_ft_lr2e5_resume500_bt256ga4`.

Why:

- it achieved the highest speaker-similarity score
- it sounded closer to the target speaker than the LoRA variants
- `checkpoint-500 + num_step=16` was the most stable baseline condition

Current judgment:

- recommended: `full_ft_lr2e5_resume500_bt256ga4`
- comparison only: `top128_attnmlp32_resume900`
- not recommended: `top128_attnmlp12`

At the current stage, LoRA variants are retained only for comparison.

## Notes

- Use `CUDA_VISIBLE_DEVICES` if you want to pin a physical GPU
- `omnivoice-kit` does not require OpenVoice as a runtime dependency
- The upstream OmniVoice source is vendored as a submodule under `third_party/OmniVoice`

## License

`omnivoice-kit` uses the same license as the bundled OmniVoice source: `Apache-2.0`.

- [LICENSE](./LICENSE)
- upstream OmniVoice: [third_party/OmniVoice/LICENSE](./third_party/OmniVoice/LICENSE)

## Credit

If you publish or distribute a Tsukuyomichan-like trained model, demo, or app, check the applicable attribution and usage conditions for your release context.

Short credit example:

```text
This project uses the Tsukuyomichan Corpus (CV: Rei Yumesaki).
```

For training:

- if you train on the Tsukuyomichan Corpus itself, verify the corpus-side terms separately
- do not treat runtime attribution and corpus-training permission as the same question
