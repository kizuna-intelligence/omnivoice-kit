from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import load_file as load_safetensors

from ._bootstrap import ensure_omnivoice_on_path


def load_lora_model(base_model: str, checkpoint_dir: str | Path, device: str | None = None):
    ensure_omnivoice_on_path()
    from omnivoice.models.omnivoice import OmniVoice

    model = OmniVoice.from_pretrained(base_model, attn_implementation="eager", load_asr=False)
    state_dict = load_safetensors(str(Path(checkpoint_dir).resolve() / "model.safetensors"))

    # Training currently saves full-model checkpoints via accelerate state saving.
    # Older/alternative flows may save PEFT adapter-style keys. Support both formats.
    if any("lora_" in key.lower() for key in state_dict):
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model.llm = get_peft_model(model.llm, lora_config)
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    return model.to(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))


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
