from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .config import TextOnlyLoraPreset


def write_text_only_lora_configs(
    train_manifest: str | Path,
    dev_manifest: str | Path,
    output_dir: str | Path,
    preset: TextOnlyLoraPreset | None = None,
) -> dict:
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preset = preset or TextOnlyLoraPreset()

    train_cfg = output_root / "train_config_text_only_lora.json"
    data_cfg = output_root / "data_config_text_only_lora.json"

    train_cfg.write_text(json.dumps(preset.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    data_payload = {
        "train": [{"language_id": "ja", "manifest_path": [str(Path(train_manifest).resolve())]}],
        "dev": [{"language_id": "ja", "manifest_path": [str(Path(dev_manifest).resolve())]}],
    }
    data_cfg.write_text(json.dumps(data_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"train_config": str(train_cfg), "data_config": str(data_cfg)}


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


def write_smoke_text_only_lora_configs(
    train_manifest: str | Path,
    dev_manifest: str | Path,
    output_dir: str | Path,
    steps: int = 1,
) -> dict:
    preset = TextOnlyLoraPreset(steps=steps, eval_steps=steps, save_steps=steps, logging_steps=1)
    return write_text_only_lora_configs(train_manifest, dev_manifest, output_dir, preset=preset)
