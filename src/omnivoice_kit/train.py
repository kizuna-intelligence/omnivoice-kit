from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .config import FullFinetunePreset, LoraTrainingPreset


def _write_training_files(
    train_manifest: str | Path,
    dev_manifest: str | Path,
    output_dir: str | Path,
    train_config_name: str,
    data_config_name: str,
    preset,
) -> dict:
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    train_cfg = output_root / train_config_name
    data_cfg = output_root / data_config_name

    train_cfg.write_text(json.dumps(preset.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    data_payload = {
        "train": [{"language_id": "ja", "manifest_path": [str(Path(train_manifest).resolve())]}],
        "dev": [{"language_id": "ja", "manifest_path": [str(Path(dev_manifest).resolve())]}],
    }
    data_cfg.write_text(json.dumps(data_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"train_config": str(train_cfg), "data_config": str(data_cfg)}


def write_lora_configs(
    train_manifest: str | Path,
    dev_manifest: str | Path,
    output_dir: str | Path,
    preset: LoraTrainingPreset | None = None,
) -> dict:
    return _write_training_files(
        train_manifest=train_manifest,
        dev_manifest=dev_manifest,
        output_dir=output_dir,
        train_config_name="train_config_lora.json",
        data_config_name="data_config_lora.json",
        preset=preset or LoraTrainingPreset(),
    )


def write_full_finetune_configs(
    train_manifest: str | Path,
    dev_manifest: str | Path,
    output_dir: str | Path,
    preset: FullFinetunePreset | None = None,
) -> dict:
    return _write_training_files(
        train_manifest=train_manifest,
        dev_manifest=dev_manifest,
        output_dir=output_dir,
        train_config_name="train_config_full_finetune.json",
        data_config_name="data_config_full_finetune.json",
        preset=preset or FullFinetunePreset(),
    )


def launch_omnivoice_train(
    train_config: str | Path,
    data_config: str | Path,
    output_dir: str | Path,
    gpu_ids: str | None = "0",
    num_processes: int = 1,
) -> subprocess.CompletedProcess:
    cmd = [
        "accelerate",
        "launch",
        "--num_processes",
        str(num_processes),
        "-m",
        "omnivoice_kit.train_entry",
        "--train_config",
        str(Path(train_config).resolve()),
        "--data_config",
        str(Path(data_config).resolve()),
        "--output_dir",
        str(Path(output_dir).resolve()),
    ]
    if gpu_ids:
        cmd[2:2] = ["--gpu_ids", gpu_ids]
    return subprocess.run(cmd, check=True)
