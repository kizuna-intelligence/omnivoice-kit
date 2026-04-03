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

For the processed `VOICEACTRESS100_###/*.flac + JSON` layout used in the Tsukuyomi-chan experiment:

```bash
omnivoice-kit prepare-voiceactress-jsonl \
  --dataset-dir /path/to/tsukuyomichan_processed \
  --output-dir work/tsukuyomi_voiceactress100
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
  --num-processes 1
```

When you want to pin a specific physical GPU, prefer `CUDA_VISIBLE_DEVICES` and omit `--gpu-ids`.
For example, `CUDA_VISIBLE_DEVICES=1` was used for the Tsukuyomi-chan `checkpoint-300` run.

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

## Checkpoint-300 speaker similarity

This repo was also used to train a full `checkpoint-300` text-only LoRA and to compare it
against base OmniVoice with the same short prompt reference:

- ref audio: `emo_024.wav` (`4.0s`)
- ref text: `急にそんなことを言われても、どう反応していいかわからないよ。`
- similarity backend: OpenVoice V2 tone-color embedding cosine similarity

Artifacts:

- LoRA JA: `artifacts/train300_similarity_ja_with_ref/summary.json`
- Base JA: `artifacts/base_similarity_ja_with_ref/summary.json`
- LoRA EN: `artifacts/train300_similarity_en_with_ref/summary.json`
- Base EN: `artifacts/base_similarity_en_with_ref/summary.json`

Results:

| Language | Model | Train-centroid mean | Pairwise mean |
| --- | --- | ---: | ---: |
| Japanese | Base OmniVoice | 0.9471 | 0.9640 |
| Japanese | LoRA checkpoint-300 | 0.9541 | 0.9667 |
| English | Base OmniVoice | 0.8751 | 0.8633 |
| English | LoRA checkpoint-300 | 0.9407 | 0.9544 |

Uplift from LoRA over base:

- Japanese: `+0.0071` centroid, `+0.0028` pairwise
- English: `+0.0656` centroid, `+0.0911` pairwise

The Japanese gain is small because short-reference prompting already anchors the voice well.
The English gain is much larger, which matches the earlier observation that transcript-only
LoRA helps the target speaker carry over across cross-lingual generation.

## Tsukuyomi-chan run

This repo was also used to train a `checkpoint-300` LoRA from the processed
`VOICEACTRESS100_###/*.flac + JSON` corpus stored under `tsukuyomichan_processed`.

Artifacts kept locally:

- train log: `artifacts/train_tsukuyomi_voiceactress100_300/train.log`
- generated samples: `artifacts/train_tsukuyomi_voiceactress100_300_generate/summary.json`
- sample wavs:
  - `artifacts/train_tsukuyomi_voiceactress100_300_generate/wav/01_tsukuyomi_001.wav`
  - `artifacts/train_tsukuyomi_voiceactress100_300_generate/wav/02_tsukuyomi_002.wav`
  - `artifacts/train_tsukuyomi_voiceactress100_300_generate/wav/03_tsukuyomi_003.wav`

## Notes

- OmniVoice itself is kept in `third_party/OmniVoice`.
- `launch-train` uses `omnivoice_kit.train_entry` so dataloaders are prepared through `accelerate`.
- This repo does not vendor OpenVoice. Speaker similarity evaluation should be treated as optional.
