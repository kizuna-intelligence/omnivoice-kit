from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import load_file as load_safetensors

from ._bootstrap import ensure_omnivoice_on_path


FULL_MODEL_LORA_TARGETS = {"audio_embeddings", "audio_heads", "embed_tokens"}


def _use_full_model_lora(target_modules: list[str] | None) -> bool:
    return bool(set(target_modules or []) & FULL_MODEL_LORA_TARGETS)


def _resolve_full_model_targets(model, requested_targets: list[str]) -> list[str]:
    resolved: list[str] = []
    for name, module in model.named_modules():
        if module.__class__.__name__ not in {"Linear", "Embedding"}:
            continue
        if name.startswith("audio_tokenizer."):
            continue
        if any(name == target or name.endswith(f".{target}") for target in requested_targets):
            resolved.append(name)
    return resolved


def load_model(base_model: str, checkpoint_dir: str | Path, device: str | None = None):
    ensure_omnivoice_on_path()
    from omnivoice.models.omnivoice import OmniVoice

    checkpoint_dir = Path(checkpoint_dir).resolve()
    model = OmniVoice.from_pretrained(base_model, attn_implementation="eager", load_asr=False)
    train_cfg_path = checkpoint_dir / "train_config.json"
    adapter_cfg_path = checkpoint_dir / "adapter_config.json"
    train_cfg = {}
    adapter_cfg = {}
    if train_cfg_path.exists():
        train_cfg = json.loads(train_cfg_path.read_text(encoding="utf-8"))
    if adapter_cfg_path.exists():
        adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
    target_modules = train_cfg.get(
        "lora_target_modules",
        adapter_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    )
    use_full_model_lora = _use_full_model_lora(target_modules)

    # Preferred path: adapter checkpoint saved by PEFT with adapter_config.json.
    if adapter_cfg_path.exists():
        if use_full_model_lora:
            lora_r = train_cfg.get("lora_r", adapter_cfg.get("r", 8))
            lora_alpha = train_cfg.get("lora_alpha", adapter_cfg.get("lora_alpha", 16))
            lora_dropout = train_cfg.get("lora_dropout", adapter_cfg.get("lora_dropout", 0.05))
            resolved_targets = _resolve_full_model_targets(model, target_modules)
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=resolved_targets,
            )
            model = get_peft_model(model, lora_config)
            adapter_state = load_safetensors(str(checkpoint_dir / "adapter_model.safetensors"))
            model.load_state_dict(adapter_state, strict=False)
        else:
            model.llm = PeftModel.from_pretrained(model.llm, str(checkpoint_dir), is_trainable=False)
    # Backward-compatible path for raw LoRA state dicts.
    else:
        state_path = checkpoint_dir / "model.safetensors"
        state_dict = load_safetensors(str(state_path))
        if any("lora_" in key.lower() for key in state_dict):
            lora_r = 8
            lora_alpha = 16
            lora_dropout = 0.05
            if train_cfg:
                lora_r = train_cfg.get("lora_r", lora_r)
                lora_alpha = train_cfg.get("lora_alpha", lora_alpha)
                lora_dropout = train_cfg.get("lora_dropout", lora_dropout)

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=target_modules,
            ) if use_full_model_lora else LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=target_modules,
            )
            if use_full_model_lora:
                model = get_peft_model(model, lora_config)
                full_state = {}
                for key, value in state_dict.items():
                    if key.startswith("base_model.model.") or "lora_" in key:
                        full_state[key] = value
                model.load_state_dict(full_state, strict=False)
            else:
                model.llm = get_peft_model(model.llm, lora_config)
                llm_state = {}
                for key, value in state_dict.items():
                    if key.startswith("llm."):
                        llm_state[key[len("llm."):]] = value
                    elif key.startswith("base_model.model.") or "lora_" in key:
                        llm_state[key] = value
                model.llm.load_state_dict(llm_state, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
    return model.to(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))


# Backward-compatible alias.
load_lora_model = load_model


def generate_from_jsonl(
    model,
    input_jsonl: str | Path,
    output_dir: str | Path,
    language: str,
    num_step: int = 16,
    seed_base: int = 2000,
    ref_audio: str | None = None,
    ref_text: str | None = None,
) -> dict:
    input_path = Path(input_jsonl).resolve()
    output_root = Path(output_dir).resolve()
    wav_dir = output_root / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = []
    for idx, row in enumerate(rows):
        torch.manual_seed(seed_base + idx)
        wav = model.generate(
            text=row["text"],
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            instruct=None,
            num_step=num_step,
        )
        wav = np.asarray(wav).squeeze()
        out_path = wav_dir / f"{idx + 1:02d}_{row['id']}.wav"
        sf.write(out_path, wav, model.sampling_rate)
        samples.append({"index": idx + 1, "id": row["id"], "text": row["text"], "audio_path": str(out_path)})

    payload = {
        "language": language,
        "num_step": num_step,
        "seed_base": seed_base,
        "ref_audio": str(Path(ref_audio).resolve()) if ref_audio else None,
        "ref_text": ref_text,
        "samples": samples,
    }
    (output_root / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
