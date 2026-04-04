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

## 現在の推奨設定

つくよみちゃん系 TTS では、現時点の最推奨は次です。

- 推奨: `full_ft_lr2e5_resume500_bt256ga4`
- 比較用: `top128_attnmlp32_resume900`
- 非推奨: `top128_attnmlp12`

判断理由:

- speaker similarity が最も高かった
- 実際に聞いた印象でも LoRA より寄りが強かった
- `checkpoint-500 + num_step=16` が最も安定していた

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

## ライセンス

`omnivoice-kit` は、同梱している OmniVoice と同じ `Apache-2.0` として扱います。

- [LICENSE](./LICENSE)
- [third_party/OmniVoice/LICENSE](./third_party/OmniVoice/LICENSE)

## クレジットと注意

つくよみちゃん寄りのモデルやデモを公開するときは、公開形態に応じてクレジットや利用条件を確認してください。

短い表記例:

```text
音声合成には「つくよみちゃんコーパス（CV: 夢前黎）」を使用しています。
```

学習については別です。

- つくよみちゃんコーパスそのものを学習に使う場合の可否や条件は、別途個別に確認してください
- モデル利用時のクレジットと、学習時の規約確認は同じ話ではありません
