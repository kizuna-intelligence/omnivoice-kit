from __future__ import annotations

import csv
import io
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import webdataset as wds
from transformers import AutoFeatureExtractor, HiggsAudioV2TokenizerModel


TARGET_SR = 24000


def _maybe_extract_archive(archive_path: Path, output_dir: Path) -> Path:
    extracted_root = output_dir / "mio_regenerated_ljspeech"
    metadata_path = extracted_root / "metadata.csv"
    if metadata_path.exists():
        return extracted_root
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-I", "zstd", "-xf", str(archive_path), "-C", str(output_dir)], check=True)
    return extracted_root


def prepare_ljspeech_jsonl(
    archive: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> dict:
    archive_path = Path(archive).resolve()
    output_root = Path(output_dir).resolve()
    dataset_root = _maybe_extract_archive(archive_path, output_root / "extracted")
    metadata_path = dataset_root / "metadata.csv"
    wav_dir = dataset_root / "wav"

    rows = []
    with metadata_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            utt_id, text = row[0], row[1]
            rows.append(
                {
                    "id": utt_id,
                    "text": text,
                    "audio_path": str((wav_dir / f"{utt_id}.wav").resolve()),
                    "language_id": "ja",
                }
            )

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        prefix = row["id"].split("_", 1)[0]
        grouped.setdefault(prefix, []).append(row)

    rng = random.Random(seed)
    train_rows = []
    dev_rows = []
    for group_rows in grouped.values():
        group_rows = list(group_rows)
        rng.shuffle(group_rows)
        cutoff = round(len(group_rows) * train_ratio)
        train_rows.extend(group_rows[:cutoff])
        dev_rows.extend(group_rows[cutoff:])

    train_rows.sort(key=lambda x: x["id"])
    dev_rows.sort(key=lambda x: x["id"])
    jsonl_dir = output_root / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    for name, split_rows in (("train", train_rows), ("dev", dev_rows)):
        with (jsonl_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in split_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "archive": str(archive_path),
        "dataset_root": str(dataset_root),
        "num_total": len(rows),
        "num_train": len(train_rows),
        "num_dev": len(dev_rows),
        "seed": seed,
        "train_ratio": train_ratio,
    }
    (output_root / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _load_audio(path: str) -> torch.Tensor:
    wav, sr = sf.read(path, always_2d=False)
    wav_t = torch.tensor(wav, dtype=torch.float32)
    if wav_t.ndim == 2:
        wav_t = wav_t.mean(dim=1)
    if sr != TARGET_SR:
        wav_t = torchaudio.functional.resample(wav_t.unsqueeze(0), sr, TARGET_SR).squeeze(0)
    wav_t = (wav_t / (wav_t.abs().max() + 1e-7)) * 0.9
    return wav_t


def _serialise_numpy(key: str, tokens: np.ndarray) -> dict:
    buffer = io.BytesIO()
    np.save(buffer, tokens)
    return {"__key__": key, "npy": buffer.getvalue()}


def tokenize_jsonl_to_manifest(
    input_jsonl: str | Path,
    output_dir: str | Path,
    tokenizer_path: str = "eustlb/higgs-audio-v2-tokenizer",
    device: str | None = None,
) -> dict:
    input_path = Path(input_jsonl).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tar_output = output_root / "data.tar"
    jsonl_output = output_root / "data.jsonl"
    manifest_output = output_root / "data.lst"
    use_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    feature_extractor = AutoFeatureExtractor.from_pretrained(tokenizer_path)
    tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(tokenizer_path, device_map=use_device)

    processed = 0
    total_duration = 0.0
    with input_path.open("r", encoding="utf-8") as f_in, wds.TarWriter(str(tar_output)) as tar_writer, jsonl_output.open("w", encoding="utf-8") as f_meta:
        for line in f_in:
            row = json.loads(line)
            wav = _load_audio(row["audio_path"])
            inputs = feature_extractor(raw_audio=wav.numpy(), sampling_rate=TARGET_SR, return_tensors="pt").to(tokenizer.device)
            with torch.inference_mode():
                audio_tokens = tokenizer.encode(inputs["input_values"]).audio_codes.squeeze(0)
            audio_tokens_np = audio_tokens.to(torch.int16).cpu().numpy()
            row["audio_duration"] = wav.numel() / TARGET_SR
            row["num_tokens"] = int(audio_tokens.size(1))
            tar_writer.write(_serialise_numpy(row["id"], audio_tokens_np))
            f_meta.write(json.dumps(row, ensure_ascii=False) + "\n")
            processed += 1
            total_duration += row["audio_duration"]

    manifest_output.write_text(f"{tar_output} {jsonl_output} {processed} {total_duration:.3f}\n", encoding="utf-8")
    return {
        "processed": processed,
        "tar_output": str(tar_output),
        "jsonl_output": str(jsonl_output),
        "manifest_output": str(manifest_output),
        "total_duration_sec": round(total_duration, 3),
    }

