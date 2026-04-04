from __future__ import annotations

import argparse
import logging
import math

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from omnivoice.training.builder import build_dataloaders, build_model_and_tokenizer
from omnivoice.training.config import TrainingConfig
from omnivoice.training.trainer import OmniTrainer

logger = logging.getLogger(__name__)

FULL_MODEL_LORA_TARGETS = {"audio_embeddings", "audio_heads", "embed_tokens"}


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


class KitTrainer(OmniTrainer):
    """Prepare dataloaders as well as model objects before training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.eval_dataloader is None:
            (self.train_dataloader,) = self.accelerator.prepare(self.train_dataloader)
        else:
            self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(
                self.train_dataloader,
                self.eval_dataloader,
            )

    def create_optimizer_and_scheduler(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        if self.config.warmup_type == "ratio":
            final_warmup_steps = math.ceil(self.config.steps * self.config.warmup_ratio)
        else:
            final_warmup_steps = self.config.warmup_steps

        if self.config.lr_scheduler_type == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=final_warmup_steps,
            )
        else:
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=final_warmup_steps,
                num_training_steps=self.config.steps,
            )
        return optimizer, lr_scheduler


def maybe_apply_lora(model, config: TrainingConfig):
    if not getattr(config, "use_lora", False):
        return model

    # Freeze the full OmniVoice stack first, then expose only PEFT adapter params.
    for param in model.parameters():
        param.requires_grad = False

    target_modules = set(config.lora_target_modules)
    if target_modules & FULL_MODEL_LORA_TARGETS:
        resolved_targets = _resolve_full_model_targets(model, list(config.lora_target_modules))
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            target_modules=resolved_targets,
        )
        model = get_peft_model(model, lora_config)
        scope = "full-model"
    else:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=list(config.lora_target_modules),
        )
        model.llm = get_peft_model(model.llm, lora_config)
        scope = "llm-only"

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Enabled PEFT LoRA (%s): trainable_params=%d total_params=%d ratio=%.6f",
        scope,
        trainable,
        total,
        trainable / total if total else 0.0,
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="omnivoice-kit training entry point")
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_config", type=str, required=True)
    args = parser.parse_args()

    config = TrainingConfig.from_json(args.train_config)
    config.output_dir = args.output_dir
    config.data_config = args.data_config

    model, tokenizer = build_model_and_tokenizer(config)
    model = maybe_apply_lora(model, config)
    train_loader, eval_loader = build_dataloaders(config, tokenizer)

    trainer = KitTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        tokenizer=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
