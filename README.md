# omnivoice-kit

OmniVoice を使って、`学習` と `推論` だけを行うための小さなキットです。

- English README: [README_en.md](./README_en.md)

このリポジトリでやることは 3 つだけです。

- LoRA 学習
- full finetune 学習
- 学習済み checkpoint / adapter を使った推論

## 最初に読む

初見でやることはこの 2 パターンです。

1. 既存モデルを使って音を出したい  
   まずは [推論](#推論いちばん早い始め方) だけ見れば足ります。
2. 自分のコーパスで学習したい  
   [学習の全体像](#学習の全体像) から順に進めてください。

## どっちを使うか

- `LoRA`
  まず軽く試したいとき向けです。比較用として残しています。
- `full finetune`
  声質をしっかり寄せたいとき向けです。現時点の推奨はこちらです。

## このリポジトリの特徴

OmniVoice 公式の学習機能をそのまま包んでいるだけではなく、`特定キャラクターの声に寄せる` ことを主眼にした構成にしています。

特に違いとして大きいのは次です。

- 学習は `ref_audio` ありきではなく、`ref_audio なし` のコーパス学習を前提にしています
- まず `素の状態でもそのキャラクターっぽい声が出る` ことを重視しています
- つまり、単なる voice cloning より `特定キャラクター特化の TTS` を作る方向に寄せています

## 現在の推奨設定

つくよみちゃん系 TTS では、現時点の最推奨は次です。

- 推奨: `full_ft_lr2e5_resume500_bt256ga4`
- 比較用: `top128_attnmlp32_resume900`
- 非推奨: `top128_attnmlp12`

判断理由:

- speaker similarity が最も高かった
- 実際に聞いた印象でも LoRA より寄りが強かった
- `checkpoint-500 + num_step=16` が最も安定していた

## 実行環境の目安

今回の推奨設定は、次のような単一 GPU 環境で実行確認しています。

- full finetune 推奨設定
  `full_ft_lr2e5_resume500_bt256ga4`
- mixed precision
  `bf16`
- 実際に使った GPU
  `RTX 5060 Ti 16GB`
- 追加確認済み GPU
  `RTX 3090 24GB`

目安:

- `16GB VRAM` 以上あれば、今回の推奨 full finetune を 1 GPU で回せる見込みが高いです
- `24GB VRAM` あると、保存時や並行作業を含めてかなり余裕があります
- `12GB VRAM` 以下では、この設定の full finetune は厳しい可能性が高いです

推論だけなら学習よりかなり軽いです。

- 推奨 checkpoint の推論
  `num_step=16`
- 目安
  `16GB VRAM` で問題なく実行できました
- 速度より表現を少し優先するなら
  `num_step=24`

## 構成

```text
third_party/OmniVoice/   OmniVoice 本体
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

## 推論: いちばん早い始め方

学習済み checkpoint か LoRA adapter があるなら、これで音が出ます。

```bash
omnivoice-kit generate \
  --base-model k2-fsa/OmniVoice \
  --checkpoint-dir /path/to/checkpoint-or-adapter \
  --input-jsonl examples/japanese_prompts.jsonl \
  --output-dir artifacts/generate_ja \
  --language ja \
  --num-step 16
```

ポイント:

- LoRA adapter でも full checkpoint でも同じ `generate` を使います
- つくよみちゃん系では `num_step=16` を基準にしています

## 学習の全体像

やることは 4 段です。

1. JSONL を作る
2. token manifest を作る
3. 学習設定を書く
4. 学習を実行する

### 1. JSONL を作る

`VOICEACTRESS100_###/*.flac + JSON` 形式ならこれです。

```bash
omnivoice-kit prepare-voiceactress-jsonl \
  --dataset-dir /path/to/tsukuyomichan_processed \
  --output-dir work/tsukuyomi_voiceactress100
```

LJSpeech 形式アーカイブならこれです。

```bash
omnivoice-kit prepare-jsonl \
  --archive /path/to/dataset.tar.zst \
  --output-dir work/dataset
```

### 2. token manifest を作る

```bash
omnivoice-kit tokenize-jsonl \
  --input-jsonl work/tsukuyomi_voiceactress100/jsonl/train.jsonl \
  --output-dir work/tokens/train

omnivoice-kit tokenize-jsonl \
  --input-jsonl work/tsukuyomi_voiceactress100/jsonl/dev.jsonl \
  --output-dir work/tokens/dev
```

### 3. 学習設定を書く

LoRA:

```bash
omnivoice-kit write-lora-configs \
  --train-manifest work/tokens/train/data.lst \
  --dev-manifest work/tokens/dev/data.lst \
  --output-dir configs/lora
```

full finetune:

```bash
omnivoice-kit write-full-finetune-configs \
  --train-manifest work/tokens/train/data.lst \
  --dev-manifest work/tokens/dev/data.lst \
  --output-dir configs/full_finetune
```

### 4. 学習を実行する

LoRA:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
omnivoice-kit launch-train \
  --train-config configs/lora/train_config_lora.json \
  --data-config configs/lora/data_config_lora.json \
  --output-dir artifacts/train_lora \
  --num-processes 1
```

full finetune:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
omnivoice-kit launch-train \
  --train-config configs/full_finetune/train_config_full_finetune.json \
  --data-config configs/full_finetune/data_config_full_finetune.json \
  --output-dir artifacts/train_full_finetune \
  --num-processes 1
```

## Submodule について

OmniVoice 本体は submodule です。

- 配置先: `third_party/OmniVoice`
- 現在の参照先: `kizuna-intelligence/OmniVoice` fork

clone 直後は必ずこれを実行してください。

```bash
git submodule update --init --recursive
```

## 低VRAM向け最適化

4 GB VRAM など低 VRAM 環境向けの最適化手順です。

### 仕組み

- **LM 圧縮 (compress-lm)**: LLM バックボーン（Qwen3-0.6B ベース）を OneCompression AutoBit で GPTQ 8bit 圧縮します。2.45 GB → 0.73 GB。
- **エンコーダー除去 (--strip-audio-encoder)**: no-ref 推論には不要な audio tokenizer のエンコーダー部分（~715 MB）を除去します。

### セットアップ

```bash
# Python 3.12 が必要
pip install "omnivoice-kit[compress]"
```

### 手順

Step 1: LM 圧縮（一度だけ実行、~8 GB VRAM 必要）

```bash
CUDA_VISIBLE_DEVICES=0 \
omnivoice-kit compress-lm \
  --model kizuna-intelligence/tsukuyomichan-omnivoice-full-finetune \
  --output-dir artifacts/compressed_lm \
  --total-budget-gb 3.0
```

Step 2: 推論（~1.4 GB VRAM、4 GB 環境で動作確認済み）

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

### VRAM 比較

| 構成 | VRAM |
|---|---|
| FP16 オリジナル | ~3.3 GB |
| LM 8bit + tokenizer FP16 | ~1.83 GB |
| LM 8bit + encoder 除去 | ~1.35 GB |

## ライセンス

`omnivoice-kit` は、同梱している OmniVoice と同じ `Apache-2.0` として扱います。

- [LICENSE](./LICENSE)
- [third_party/OmniVoice/LICENSE](./third_party/OmniVoice/LICENSE)

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
