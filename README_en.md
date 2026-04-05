# omnivoice-kit

General-purpose training and inference kit for OmniVoice. Train on any corpus and build a TTS specialized for your target speaker or character.

- 日本語 README: [README.md](./README.md)

Pre-trained models:

| Model | Speaker | HuggingFace |
|---|---|---|
| Tsukuyomichan (compressed) | Tsukuyomichan (© Rei Yumesaki) | [kizuna-intelligence/tsukuyomichan-omnivoice-compressed](https://huggingface.co/kizuna-intelligence/tsukuyomichan-omnivoice-compressed) |
| Amitaro ITA (normal) | Amitaro Voice Material Studio | [kizuna-intelligence/amitaro-ita-omnivoice-full-finetune](https://huggingface.co/kizuna-intelligence/amitaro-ita-omnivoice-full-finetune) |
| Sayoko | Fusic (81-year-old female) | [kizuna-intelligence/sayoko-omnivoice-full-finetune](https://huggingface.co/kizuna-intelligence/sayoko-omnivoice-full-finetune) |

> **Note:** Before using any voice model, please refer to the terms of use listed on each model's page.

This kit supports three things:

1. Synthesize audio with a pre-trained model (inference)
2. LoRA fine-tuning on your own corpus
3. Full fine-tuning on your own corpus

---

## Pre-trained Models

### Tsukuyomichan

A pre-trained Tsukuyomichan model is available on HuggingFace in two variants.

| Variant | Folder | VRAM | RTF | Use Case |
|---|---|---|---|---|
| **FP16 (standard)** | `…/fp16` | ~3.3 GB | 0.11 | Normal inference · high quality |
| **GPTQ 8-bit (compressed)** | `…/gptq8` | ~1.35 GB | 0.34 | Low-VRAM devices (4 GB GPU) |

> **What is RTF?** Real-Time Factor — how many seconds it takes to generate 1 second of audio. Both variants are far below 0.5, meaning audio is generated several times faster than real-time.

Which one to use:
- **8 GB or more VRAM** → Use FP16 (standard)
- **4 – 6 GB VRAM** → Use GPTQ 8-bit (compressed)

### Amitaro ITA (normal style)

A model trained on ITA corpus recordings by Amitaro Voice Material Studio. FP16 only.

```bash
CUDA_VISIBLE_DEVICES=0 \
omnivoice-kit generate \
  --base-model kizuna-intelligence/amitaro-ita-omnivoice-full-finetune \
  --input-jsonl examples/japanese_prompts.jsonl \
  --output-dir artifacts/generate_amitaro \
  --language ja \
  --num-step 20
```

**License / Terms of use (required reading):** [https://amitaro.net/voice/ita/](https://amitaro.net/voice/ita/)

When publishing anything made with this model, you **must** include the following credit:

```
あみたろの声素材工房（https://amitaro.net/）
```

### Sayoko

A model trained on the Fusic Sayoko voice corpus (CC BY 4.0). The speaker is an 81-year-old female.

```bash
CUDA_VISIBLE_DEVICES=0 \
omnivoice-kit generate \
  --base-model kizuna-intelligence/sayoko-omnivoice-full-finetune \
  --input-jsonl examples/japanese_prompts.jsonl \
  --output-dir artifacts/generate_sayoko \
  --language ja \
  --num-step 20
```

**License:** CC BY 4.0

When publishing anything made with this model, you **must** include the following credit:

```
Fusic/サヨ子音声コーパス（https://huggingface.co/datasets/bandad/sayoko-tts-corpus）
```

---

## Setup

```bash
git clone https://github.com/kizuna-intelligence/omnivoice-kit
cd omnivoice-kit
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate
pip install -e third_party/OmniVoice
pip install -e .
```

Install extras based on your use case:

| Use case | Command | Python |
|---|---|---|
| FP16 inference only | `pip install -e .` | 3.10+ |
| GPTQ compressed model inference | `pip install -e ".[gptq]"` | **3.12 required** |
| Compress a model yourself (compress-lm) | `pip install -e ".[compress]"` | **3.12 required** |

---

## Inference

### FP16 (standard)

```bash
CUDA_VISIBLE_DEVICES=0 \
omnivoice-kit generate \
  --base-model kizuna-intelligence/tsukuyomichan-omnivoice-full-finetune \
  --checkpoint-dir kizuna-intelligence/tsukuyomichan-omnivoice-compressed/fp16 \
  --input-jsonl examples/japanese_prompts.jsonl \
  --output-dir artifacts/generate_fp16 \
  --language ja \
  --num-step 16
```

### GPTQ 8-bit (compressed, for 4 GB GPU)

```bash
CUDA_VISIBLE_DEVICES=0 \
omnivoice-kit generate \
  --base-model kizuna-intelligence/tsukuyomichan-omnivoice-full-finetune \
  --checkpoint-dir kizuna-intelligence/tsukuyomichan-omnivoice-compressed/gptq8 \
  --strip-audio-encoder \
  --input-jsonl examples/japanese_prompts.jsonl \
  --output-dir artifacts/generate_gptq8 \
  --language ja \
  --num-step 16
```

`--strip-audio-encoder` removes the audio encoder modules (~715 MB) that are unused during no-ref inference. This saves significant VRAM.

### Amitaro ITA model

```bash
CUDA_VISIBLE_DEVICES=0 \
omnivoice-kit generate \
  --base-model kizuna-intelligence/amitaro-ita-omnivoice-full-finetune \
  --input-jsonl examples/japanese_prompts.jsonl \
  --output-dir artifacts/generate_amitaro \
  --language ja \
  --num-step 20
```

**Terms of use (required reading):** [https://amitaro.net/voice/ita/](https://amitaro.net/voice/ita/)

### Sayoko model

```bash
CUDA_VISIBLE_DEVICES=0 \
omnivoice-kit generate \
  --base-model kizuna-intelligence/sayoko-omnivoice-full-finetune \
  --input-jsonl examples/japanese_prompts.jsonl \
  --output-dir artifacts/generate_sayoko \
  --language ja \
  --num-step 20
```

**License:** CC BY 4.0 (see [model page](https://huggingface.co/kizuna-intelligence/sayoko-omnivoice-full-finetune))

### Input File Format

Prepare a JSONL file with one JSON object per line, as in `examples/japanese_prompts.jsonl`:

```json
{"id": "001", "text": "こんにちは、今日はいい天気ですね。"}
{"id": "002", "text": "やった！ようやく完成したよ！"}
```

---

## Low-VRAM: Compress the Model Yourself

A pre-compressed model is already available on [HuggingFace](https://huggingface.co/kizuna-intelligence/tsukuyomichan-omnivoice-compressed). If you want to compress a different model, follow these steps.

**Step 1: Compress the LLM (one-time, needs ~8 GB VRAM)**

```bash
CUDA_VISIBLE_DEVICES=0 \
omnivoice-kit compress-lm \
  --model kizuna-intelligence/tsukuyomichan-omnivoice-full-finetune \
  --output-dir artifacts/compressed_lm \
  --total-budget-gb 3.0
```

**Step 2: Run inference with the compressed model**

```bash
CUDA_VISIBLE_DEVICES=0 \
omnivoice-kit generate \
  --base-model kizuna-intelligence/tsukuyomichan-omnivoice-full-finetune \
  --checkpoint-dir artifacts/compressed_lm \
  --strip-audio-encoder \
  --input-jsonl examples/japanese_prompts.jsonl \
  --output-dir artifacts/generate_low_vram \
  --language ja
```

---

## Training on Your Own Corpus

### Step 1: Prepare data

**`VOICEACTRESS100_###/*.flac + JSON` format**

```bash
omnivoice-kit prepare-voiceactress-jsonl \
  --dataset-dir /path/to/tsukuyomichan_processed \
  --output-dir work/tsukuyomi_voiceactress100
```

**LJSpeech-style archive**

```bash
omnivoice-kit prepare-jsonl \
  --archive /path/to/dataset.tar.zst \
  --output-dir work/dataset
```

### Step 2: Tokenize

```bash
omnivoice-kit tokenize-jsonl \
  --input-jsonl work/tsukuyomi_voiceactress100/jsonl/train.jsonl \
  --output-dir work/tokens/train

omnivoice-kit tokenize-jsonl \
  --input-jsonl work/tsukuyomi_voiceactress100/jsonl/dev.jsonl \
  --output-dir work/tokens/dev
```

### Step 3: Generate config files

**For LoRA**

```bash
omnivoice-kit write-lora-configs \
  --train-manifest work/tokens/train/data.lst \
  --dev-manifest work/tokens/dev/data.lst \
  --output-dir configs/lora
```

**For Full Finetune (recommended)**

```bash
omnivoice-kit write-full-finetune-configs \
  --train-manifest work/tokens/train/data.lst \
  --dev-manifest work/tokens/dev/data.lst \
  --output-dir configs/full_finetune
```

### Step 4: Run training

**LoRA**

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
omnivoice-kit launch-train \
  --train-config configs/lora/train_config_lora.json \
  --data-config configs/lora/data_config_lora.json \
  --output-dir artifacts/train_lora \
  --num-processes 1
```

**Full Finetune**

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
omnivoice-kit launch-train \
  --train-config configs/full_finetune/train_config_full_finetune.json \
  --data-config configs/full_finetune/data_config_full_finetune.json \
  --output-dir artifacts/train_full_finetune \
  --num-processes 1
```

### LoRA vs Full Finetune

- **LoRA**: Lighter. Kept for comparison purposes.
- **Full Finetune**: Higher voice similarity. Recommended when building a TTS for a specific character.

### VRAM Requirements for Training

| Task | Minimum VRAM | Recommended VRAM |
|---|---|---|
| Full finetune | 16 GB | 24 GB |
| Inference only | 3.3 GB (FP16) / 1.35 GB (compressed) | — |

---

## About This Repository

This kit is designed not just to run OmniVoice out of the box, but to build **character-specialized TTS** for a target speaker:

- Training does not require reference audio (`ref_audio`)
- Inference is optimized for no-ref output that sounds like the target character
- The focus is on speaker specialization, not general-purpose voice cloning

```text
third_party/OmniVoice/   OmniVoice upstream (submodule)
src/omnivoice_kit/       Training, inference, and optimization wrappers
examples/                Sample input files
```

---

## License

`omnivoice-kit` uses the same license as OmniVoice: `Apache-2.0`.

- [LICENSE](./LICENSE)
- [third_party/OmniVoice/LICENSE](./third_party/OmniVoice/LICENSE)

---

## Credit and Usage Conditions

### Tsukuyomichan Corpus Attribution

The Tsukuyomichan model included in this toolkit was trained using voice data freely released by the free-use character "Tsukuyomichan" (© Rei Yumesaki).

When publishing a Tsukuyomichan-based model, demo, or application, include the following credit in full. Please include the URL.

```text
This software uses voice data freely released by the free-use character "Tsukuyomichan" (© Rei Yumesaki).

■ Tsukuyomichan Corpus (CV: Rei Yumesaki)
https://tyc.rei-yumesaki.net/material/corpus/
```

Short credit example for end users:

```text
This project uses the Tsukuyomichan Corpus (CV: Rei Yumesaki).
```

### Restrictions on Output Audio

Audio generated from the Tsukuyomichan model must not be used for the following purposes:

- Criticizing or attacking individuals. (The definition of "criticism or attack" follows the [Tsukuyomichan Character License](https://tyc.rei-yumesaki.net/about/terms/#condition3).)
- Advocating for or against specific political positions, religions, or ideologies.
- Publishing explicit or offensive content without appropriate age-gating.
- Distributing or publishing in a form that permits secondary use (use as raw material) by others.

※ Distribution or sale as finished creative works is permitted.

### Modification and Redistribution

If you use the Tsukuyomichan model itself as a base (including modification, fine-tuning, merging with other models, or redistribution), the portions derived from the Tsukuyomichan Corpus must be handled in accordance with the [Tsukuyomichan Corpus Terms of Use](https://tyc.rei-yumesaki.net/material/corpus/). This requirement is copyleft and carries over to all derivative works and redistributed data.
