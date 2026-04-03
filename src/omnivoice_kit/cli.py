from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import prepare_ljspeech_jsonl, prepare_voiceactress100_jsonl, tokenize_jsonl_to_manifest
from .infer import generate_from_jsonl, load_lora_model
from .train import launch_omnivoice_train, write_smoke_text_only_lora_configs, write_text_only_lora_configs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omnivoice-kit")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare-jsonl")
    p_prepare.add_argument("--archive", required=True)
    p_prepare.add_argument("--output-dir", required=True)
    p_prepare.add_argument("--train-ratio", type=float, default=0.9)
    p_prepare.add_argument("--seed", type=int, default=42)

    p_prepare_va = sub.add_parser("prepare-voiceactress-jsonl")
    p_prepare_va.add_argument("--dataset-dir", required=True)
    p_prepare_va.add_argument("--output-dir", required=True)
    p_prepare_va.add_argument("--train-ratio", type=float, default=0.9)
    p_prepare_va.add_argument("--seed", type=int, default=42)

    p_tok = sub.add_parser("tokenize-jsonl")
    p_tok.add_argument("--input-jsonl", required=True)
    p_tok.add_argument("--output-dir", required=True)
    p_tok.add_argument("--tokenizer-path", default="eustlb/higgs-audio-v2-tokenizer")
    p_tok.add_argument("--device", default=None)

    p_cfg = sub.add_parser("write-text-only-configs")
    p_cfg.add_argument("--train-manifest", required=True)
    p_cfg.add_argument("--dev-manifest", required=True)
    p_cfg.add_argument("--output-dir", required=True)
    p_cfg.add_argument("--steps", type=int, default=None)

    p_launch = sub.add_parser("launch-train")
    p_launch.add_argument("--train-config", required=True)
    p_launch.add_argument("--data-config", required=True)
    p_launch.add_argument("--output-dir", required=True)
    p_launch.add_argument("--gpu-ids", default=None)
    p_launch.add_argument("--num-processes", type=int, default=1)

    p_gen = sub.add_parser("generate")
    p_gen.add_argument("--base-model", required=True)
    p_gen.add_argument("--checkpoint-dir", required=True)
    p_gen.add_argument("--input-jsonl", required=True)
    p_gen.add_argument("--output-dir", required=True)
    p_gen.add_argument("--language", required=True)
    p_gen.add_argument("--num-step", type=int, default=16)
    p_gen.add_argument("--seed-base", type=int, default=2000)
    p_gen.add_argument("--ref-audio", default=None)
    p_gen.add_argument("--ref-text", default=None)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare-jsonl":
        result = prepare_ljspeech_jsonl(args.archive, args.output_dir, train_ratio=args.train_ratio, seed=args.seed)
    elif args.command == "prepare-voiceactress-jsonl":
        result = prepare_voiceactress100_jsonl(args.dataset_dir, args.output_dir, train_ratio=args.train_ratio, seed=args.seed)
    elif args.command == "tokenize-jsonl":
        result = tokenize_jsonl_to_manifest(args.input_jsonl, args.output_dir, tokenizer_path=args.tokenizer_path, device=args.device)
    elif args.command == "write-text-only-configs":
        if args.steps is None:
            result = write_text_only_lora_configs(args.train_manifest, args.dev_manifest, args.output_dir)
        else:
            result = write_smoke_text_only_lora_configs(args.train_manifest, args.dev_manifest, args.output_dir, steps=args.steps)
    elif args.command == "launch-train":
        completed = launch_omnivoice_train(
            train_config=args.train_config,
            data_config=args.data_config,
            output_dir=args.output_dir,
            gpu_ids=args.gpu_ids,
            num_processes=args.num_processes,
        )
        result = {"returncode": completed.returncode, "output_dir": str(Path(args.output_dir).resolve())}
    elif args.command == "generate":
        model = load_lora_model(args.base_model, args.checkpoint_dir)
        result = generate_from_jsonl(
            model=model,
            input_jsonl=args.input_jsonl,
            output_dir=args.output_dir,
            language=args.language,
            num_step=args.num_step,
            seed_base=args.seed_base,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
        )
    else:
        raise ValueError(args.command)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
