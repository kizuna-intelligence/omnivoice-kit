from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class BaseTrainingPreset:
    llm_name_or_path: str = "Qwen/Qwen3-0.6B"
    audio_vocab_size: int = 1025
    audio_mask_id: int = 1024
    num_audio_codebook: int = 8
    audio_codebook_weights: list[int] = field(default_factory=lambda: [8, 8, 6, 6, 4, 4, 2, 2])
    drop_cond_ratio: float = 0.0
    prompt_ratio_range: list[float] = field(default_factory=lambda: [0.0, 0.0])
    mask_ratio_range: list[float] = field(default_factory=lambda: [0.2, 1.0])
    language_ratio: float = 1.0
    use_pinyin_ratio: float = 0.0
    instruct_ratio: float = 0.0
    only_instruct_ratio: float = 0.0
    resume_from_checkpoint: str | None = None
    init_from_checkpoint: str | None = "k2-fsa/OmniVoice"
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    steps: int = 300
    seed: int = 42
    lr_scheduler_type: str = "cosine"
    warmup_type: str = "ratio"
    warmup_ratio: float = 0.03
    warmup_steps: int = 0
    batch_tokens: int = 1024
    gradient_accumulation_steps: int = 1
    num_workers: int = 1
    mixed_precision: str = "bf16"
    allow_tf32: bool = True
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 250
    keep_last_n_checkpoints: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LoraTrainingPreset(BaseTrainingPreset):
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])


@dataclass
class FullFinetunePreset(BaseTrainingPreset):
    use_lora: bool = False


# Backward-compatible alias.
TextOnlyLoraPreset = LoraTrainingPreset
