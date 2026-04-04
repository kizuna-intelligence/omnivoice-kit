# omnivoice-kit

OmniVoice のための最小限の学習・推論キットです。

このリポジトリでは、扱う対象を次の 3 つに絞っています。

- LoRA 学習
- full finetune 学習
- 学習済み checkpoint / adapter を使った推論

実験用の動画生成や個別検証コードは、このリポジトリには置かない方針です。

## 構成

```text
third_party/OmniVoice/   上流 OmniVoice
src/omnivoice_kit/       学習と推論の薄いラッパー
examples/                最低限の入力例
```

## セットアップ

```bash
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate
pip install -e third_party/OmniVoice
pip install -e .
```

## データ準備

LJSpeech 形式アーカイブ:

```bash
omnivoice-kit prepare-jsonl \
  --archive /path/to/dataset.tar.zst \
  --output-dir work/dataset
```

`VOICEACTRESS100_###/*.flac + JSON` 形式:

```bash
omnivoice-kit prepare-voiceactress-jsonl \
  --dataset-dir /path/to/tsukuyomichan_processed \
  --output-dir work/tsukuyomi_voiceactress100
```

manifest 生成:

```bash
omnivoice-kit tokenize-jsonl \
  --input-jsonl work/tsukuyomi_voiceactress100/jsonl/train.jsonl \
  --output-dir work/tokens/train

omnivoice-kit tokenize-jsonl \
  --input-jsonl work/tsukuyomi_voiceactress100/jsonl/dev.jsonl \
  --output-dir work/tokens/dev
```

## LoRA 学習

設定ファイル生成:

```bash
omnivoice-kit write-lora-configs \
  --train-manifest work/tokens/train/data.lst \
  --dev-manifest work/tokens/dev/data.lst \
  --output-dir configs/lora
```

学習:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
omnivoice-kit launch-train \
  --train-config configs/lora/train_config_lora.json \
  --data-config configs/lora/data_config_lora.json \
  --output-dir artifacts/train_lora \
  --num-processes 1
```

## Full Finetune 学習

設定ファイル生成:

```bash
omnivoice-kit write-full-finetune-configs \
  --train-manifest work/tokens/train/data.lst \
  --dev-manifest work/tokens/dev/data.lst \
  --output-dir configs/full_finetune
```

学習:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
omnivoice-kit launch-train \
  --train-config configs/full_finetune/train_config_full_finetune.json \
  --data-config configs/full_finetune/data_config_full_finetune.json \
  --output-dir artifacts/train_full_finetune \
  --num-processes 1
```

## 推論

LoRA adapter でも full checkpoint でも、同じ `generate` で読めます。

```bash
omnivoice-kit generate \
  --base-model k2-fsa/OmniVoice \
  --checkpoint-dir /path/to/checkpoint-or-adapter \
  --input-jsonl examples/japanese_prompts.jsonl \
  --output-dir artifacts/generate_ja \
  --language ja \
  --num-step 16
```

## つくよみちゃん TTS のおすすめ

現時点での最推奨は、`full_ft_lr2e5_resume500_bt256ga4` のフルファインチューニング設定です。

理由:

- 話者類似度の数値が最も高い
- 実際に聞いた印象でも、LoRA より声質の寄りが強い
- `checkpoint-500 + num_step=16` が基準条件として最も安定していた

判断:

- 最推奨: `full_ft_lr2e5_resume500_bt256ga4`
- 比較用: `top128_attnmlp32_resume900`
- 非推奨: `top128_attnmlp12`

LoRA は現時点では比較用です。`r=32` は声質寄りですが最終推奨ではなく、`r=12` も表現寄り比較として残しているだけで、推奨設定にはしません。

## 補足

- 物理 GPU を固定したい場合は `CUDA_VISIBLE_DEVICES` を使ってください
- `omnivoice-kit` は OpenVoice を必須依存にはしていません
- 上流 OmniVoice 本体は `third_party/OmniVoice` に置いています
