# omnivoice-kit

OmniVoice を使って特定キャラクターの音声合成モデルを学習・推論するための汎用キットです。任意のコーパスで学習でき、特定の話者に特化した TTS を構築できます。

- English README: [README_en.md](./README_en.md)

公開済みの学習済みモデル：

| モデル | 話者 | HuggingFace |
|---|---|---|
| つくよみちゃん（圧縮版） | つくよみちゃん（© Rei Yumesaki） | [kizuna-intelligence/tsukuyomichan-omnivoice-compressed](https://huggingface.co/kizuna-intelligence/tsukuyomichan-omnivoice-compressed) |
| あみたろ ITA（通常） | あみたろの声素材工房 | [kizuna-intelligence/amitaro-ita-omnivoice-full-finetune](https://huggingface.co/kizuna-intelligence/amitaro-ita-omnivoice-full-finetune) |
| サヨ子 | Fusic（81 歳女性） | [kizuna-intelligence/sayoko-omnivoice-full-finetune](https://huggingface.co/kizuna-intelligence/sayoko-omnivoice-full-finetune) |

> **注意：** 各音声モデルを利用する前に、各モデルページに記載の利用規約を必ず参照してください。

できることは 3 つです。

1. 学習済みモデルで音声を合成する（推論）
2. 自分のコーパスで LoRA 学習する
3. 自分のコーパスで full finetune する

---

## まず音を出したい人へ

### つくよみちゃんモデル

つくよみちゃんの学習済みモデルを公開しています。HuggingFace に 2 種類用意されています。

| バリアント | フォルダ | 必要 VRAM | 速さ (RTF) | 用途 |
|---|---|---|---|---|
| **FP16（通常版）** | `…/fp16` | ~3.3 GB | 0.11 | 通常の推論・高品質 |
| **GPTQ 8-bit（圧縮版）** | `…/gptq8` | ~1.35 GB | 0.34 | 4 GB GPU など低 VRAM 環境 |

> **RTF とは？** 音声 1 秒を生成するのにかかる時間です。どちらも 0.5 を大きく下回っており、リアルタイムより数倍速く生成できます。

どちらを使うか迷ったら：
- **8 GB 以上の GPU がある** → FP16（通常版）を使ってください
- **4 〜 6 GB の GPU しかない** → GPTQ 8-bit（圧縮版）を使ってください

### あみたろ ITA モデル（通常スタイル）

あみたろの声素材工房の ITA コーパス読み上げ音声で学習したモデルです（FP16 のみ）。

```bash
CUDA_VISIBLE_DEVICES=0 \
omnivoice-kit generate \
  --base-model kizuna-intelligence/amitaro-ita-omnivoice-full-finetune \
  --input-jsonl examples/japanese_prompts.jsonl \
  --output-dir artifacts/generate_amitaro \
  --language ja \
  --num-step 20
```

**利用規約（必読）：** [https://amitaro.net/voice/ita/](https://amitaro.net/voice/ita/)

本モデルを使用した成果物には、以下のクレジット表記が**必須**です：

```
あみたろの声素材工房（https://amitaro.net/）
```

### サヨ子モデル

Fusic のサヨ子音声コーパス（CC BY 4.0）で学習した 81 歳女性の声モデルです。

```bash
CUDA_VISIBLE_DEVICES=0 \
omnivoice-kit generate \
  --base-model kizuna-intelligence/sayoko-omnivoice-full-finetune \
  --input-jsonl examples/japanese_prompts.jsonl \
  --output-dir artifacts/generate_sayoko \
  --language ja \
  --num-step 20
```

**ライセンス：** CC BY 4.0

本モデルを使用した成果物には、以下のクレジット表記が**必須**です：

```
Fusic/サヨ子音声コーパス（https://huggingface.co/datasets/bandad/sayoko-tts-corpus）
```

---

## セットアップ

```bash
git clone https://github.com/kizuna-intelligence/omnivoice-kit
cd omnivoice-kit
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate
pip install -e third_party/OmniVoice
pip install -e .
```

用途に応じて追加インストール：

| 用途 | コマンド | Python |
|---|---|---|
| FP16 推論のみ | `pip install -e .` | 3.10+ |
| GPTQ 圧縮モデルで推論 | `pip install -e ".[gptq]"` | **3.12 必須** |
| モデルを自分で圧縮（compress-lm） | `pip install -e ".[compress]"` | **3.12 必須** |

---

## 推論

### FP16（通常版）で音を出す

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

### GPTQ 8-bit（圧縮版）で音を出す（4 GB GPU 向け）

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

`--strip-audio-encoder` は、no-ref 推論では使わない音声エンコーダー（約 715 MB）を除去するオプションです。VRAM の節約になります。

### あみたろ ITA モデルで音を出す

```bash
CUDA_VISIBLE_DEVICES=0 \
omnivoice-kit generate \
  --base-model kizuna-intelligence/amitaro-ita-omnivoice-full-finetune \
  --input-jsonl examples/japanese_prompts.jsonl \
  --output-dir artifacts/generate_amitaro \
  --language ja \
  --num-step 20
```

**利用規約（必読）：** [https://amitaro.net/voice/ita/](https://amitaro.net/voice/ita/)

### サヨ子モデルで音を出す

```bash
CUDA_VISIBLE_DEVICES=0 \
omnivoice-kit generate \
  --base-model kizuna-intelligence/sayoko-omnivoice-full-finetune \
  --input-jsonl examples/japanese_prompts.jsonl \
  --output-dir artifacts/generate_sayoko \
  --language ja \
  --num-step 20
```

**ライセンス：** CC BY 4.0（[モデルページ参照](https://huggingface.co/kizuna-intelligence/sayoko-omnivoice-full-finetune)）

### 入力ファイルの形式

`examples/japanese_prompts.jsonl` のように 1 行 1 件の JSON で用意します。

```json
{"id": "001", "text": "こんにちは、今日はいい天気ですね。"}
{"id": "002", "text": "やった！ようやく完成したよ！"}
```

---

## 低 VRAM 向け：自分でモデルを圧縮する

すでに圧縮済みモデルが [HuggingFace](https://huggingface.co/kizuna-intelligence/tsukuyomichan-omnivoice-compressed) にあります。別のモデルを圧縮したい場合は以下の手順で行います。

**Step 1: LM を圧縮する（一度だけ実行、~8 GB VRAM 必要）**

```bash
CUDA_VISIBLE_DEVICES=0 \
omnivoice-kit compress-lm \
  --model kizuna-intelligence/tsukuyomichan-omnivoice-full-finetune \
  --output-dir artifacts/compressed_lm \
  --total-budget-gb 3.0
```

**Step 2: 圧縮済みモデルで推論する**

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

## 自分のコーパスで学習したい場合

### Step 1: データを準備する

**`VOICEACTRESS100_###/*.flac + JSON` 形式の場合**

```bash
omnivoice-kit prepare-voiceactress-jsonl \
  --dataset-dir /path/to/tsukuyomichan_processed \
  --output-dir work/tsukuyomi_voiceactress100
```

**LJSpeech 形式アーカイブの場合**

```bash
omnivoice-kit prepare-jsonl \
  --archive /path/to/dataset.tar.zst \
  --output-dir work/dataset
```

### Step 2: トークン変換する

```bash
omnivoice-kit tokenize-jsonl \
  --input-jsonl work/tsukuyomi_voiceactress100/jsonl/train.jsonl \
  --output-dir work/tokens/train

omnivoice-kit tokenize-jsonl \
  --input-jsonl work/tsukuyomi_voiceactress100/jsonl/dev.jsonl \
  --output-dir work/tokens/dev
```

### Step 3: 設定ファイルを生成する

**LoRA 学習の場合**

```bash
omnivoice-kit write-lora-configs \
  --train-manifest work/tokens/train/data.lst \
  --dev-manifest work/tokens/dev/data.lst \
  --output-dir configs/lora
```

**Full finetune の場合（推奨）**

```bash
omnivoice-kit write-full-finetune-configs \
  --train-manifest work/tokens/train/data.lst \
  --dev-manifest work/tokens/dev/data.lst \
  --output-dir configs/full_finetune
```

### Step 4: 学習を実行する

**LoRA**

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
omnivoice-kit launch-train \
  --train-config configs/lora/train_config_lora.json \
  --data-config configs/lora/data_config_lora.json \
  --output-dir artifacts/train_lora \
  --num-processes 1
```

**Full finetune**

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
omnivoice-kit launch-train \
  --train-config configs/full_finetune/train_config_full_finetune.json \
  --data-config configs/full_finetune/data_config_full_finetune.json \
  --output-dir artifacts/train_full_finetune \
  --num-processes 1
```

### LoRA か Full finetune か

- **LoRA**: 軽い。比較用として残しています。
- **Full finetune**: 声質の再現度が高い。特定キャラクターの TTS を作る場合はこちらを推奨します。

### 学習に必要な VRAM

| 用途 | 最低 VRAM | 推奨 VRAM |
|---|---|---|
| Full finetune | 16 GB | 24 GB |
| 推論のみ | 3.3 GB（FP16）/ 1.35 GB（圧縮版） | — |

---

## このリポジトリについて

このキットは、OmniVoice をそのまま動かすだけでなく、**特定キャラクターの声に特化した TTS** を作ることを目的に設計しています。具体的には：

- 学習時に参照音声（ref_audio）を必要としない構成になっています
- 推論時も参照なし（no-ref）でキャラクターの声が出ることを優先しています
- 汎用的な voice cloning ではなく、特定話者への特化を重視しています

```text
third_party/OmniVoice/   OmniVoice 本体（submodule）
src/omnivoice_kit/       学習・推論・最適化のラッパー
examples/                入力ファイルのサンプル
```

---

## ライセンス

`omnivoice-kit` は OmniVoice と同じ `Apache-2.0` です。

- [LICENSE](./LICENSE)
- [third_party/OmniVoice/LICENSE](./third_party/OmniVoice/LICENSE)

---

## クレジットと注意

### つくよみちゃんコーパス クレジット

本ツールで使用するつくよみちゃんモデルの学習には、フリー素材キャラクター「つくよみちゃん」（© Rei Yumesaki）が無料公開している音声データを使用しています。

つくよみちゃんモデルやデモを公開する際は、以下のクレジットを省略せず掲載してください。

```text
本ソフトウェアの音声合成には、フリー素材キャラクター「つくよみちゃん」（© Rei Yumesaki）が無料公開している音声データを使用しています。

■つくよみちゃんコーパス（CV.夢前黎）
https://tyc.rei-yumesaki.net/material/corpus/
```

ユーザーへの短い案内例:

```text
音声合成には「つくよみちゃんコーパス（CV: 夢前黎）」を使用しています。
```

### 出力音声の利用制限

つくよみちゃんモデルから出力した音声は、次の目的ではご利用いただけません。

- 人を批判・攻撃すること。（「批判・攻撃」の定義は、[つくよみちゃんキャラクターライセンス](https://tyc.rei-yumesaki.net/about/terms/#condition3) に準じます）
- 特定の政治的立場・宗教・思想への賛同または反対を呼びかけること。
- 刺激の強い表現をゾーニングなしで公開すること。
- 他者に対して二次利用（素材としての利用）を許可する形で公開すること。

※鑑賞用の作品として配布・販売していただくことは問題ございません。

### 改変・再配布について

つくよみちゃんのモデルそのものを素材として使用する場合（改変、ファインチューニング、他モデルとのマージ、再配布などを行う場合）、つくよみちゃんコーパスに由来する部分の取り扱いについては、[つくよみちゃんコーパスの利用規約](https://tyc.rei-yumesaki.net/material/corpus/) に従ってください。この規定は、派生ソフトや再配布されたデータにもコピーレフトされます。
