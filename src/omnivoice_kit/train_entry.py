from __future__ import annotations

import argparse

from omnivoice.training.builder import build_dataloaders, build_model_and_tokenizer
from omnivoice.training.config import TrainingConfig
from omnivoice.training.trainer import OmniTrainer


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
