"""Low-VRAM optimization utilities for OmniVoice models.

Two operations are provided:

1. ``compress_lm`` — extract the LLM backbone from an OmniVoice checkpoint and
   compress it with OneCompression AutoBit (8-bit GPTQ by default).
   Requires ``onecomp`` (Python >= 3.12)::

       pip install onecomp

2. ``strip_audio_encoder`` — remove the audio-encoder-only submodules from a
   loaded OmniVoice model's ``audio_tokenizer`` in-place.  Only the decoder
   path (``acoustic_decoder``, ``quantizer``, ``fc2``) is kept.  Safe to call
   whenever you are doing no-ref inference and will never call
   ``audio_tokenizer.encode()``.

Typical workflow for a 4 GB VRAM device
----------------------------------------
Step 1 (one-time, needs ~8 GB VRAM for calibration):

.. code-block:: bash

    omnivoice-kit compress-lm \\
        --model kizuna-intelligence/tsukuyomichan-omnivoice-full-finetune \\
        --output-dir artifacts/compressed_lm

Step 2 (inference, ~1.4 GB peak VRAM):

.. code-block:: bash

    omnivoice-kit generate \\
        --base-model kizuna-intelligence/tsukuyomichan-omnivoice-full-finetune \\
        --checkpoint-dir artifacts/compressed_lm \\
        --strip-audio-encoder \\
        --input-jsonl examples/japanese_prompts.jsonl \\
        --output-dir artifacts/generate_low_vram \\
        --language ja
"""
from __future__ import annotations

import gc
import json
from pathlib import Path


# ── Audio encoder stripping ──────────────────────────────────────────────────

_ENCODER_ONLY_MODULES = [
    "semantic_model",    # HuBERT  377 MB — used only by encode()
    "acoustic_encoder",  # DAC     205 MB — used only by encode()
    "encoder_semantic",  #          59 MB — used only by encode()
    "decoder_semantic",  #          66 MB — dead code, never called
    "fc",                #           4 MB — used only by encode()
    "fc1",               #           3 MB — dead code, never called
]


def strip_audio_encoder(model) -> None:
    """Remove encoder-only submodules from ``model.audio_tokenizer`` in-place.

    After this call the tokenizer can no longer encode reference audio, but
    ``decode()`` (tokens → waveform) works unchanged.  Saves ~715 MB of VRAM.

    Args:
        model: A loaded ``OmniVoice`` instance.
    """
    import torch, gc as _gc
    tok = model.audio_tokenizer
    removed: list[str] = []
    for name in _ENCODER_ONLY_MODULES:
        if hasattr(tok, name):
            delattr(tok, name)
            removed.append(name)
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if removed:
        mb = sum({
            "semantic_model": 377, "acoustic_encoder": 205, "encoder_semantic": 59,
            "decoder_semantic": 66, "fc": 4, "fc1": 3,
        }.get(n, 0) for n in removed)
        print(f"[strip_audio_encoder] removed {removed}  (~{mb} MB freed)")


# ── LM compression ───────────────────────────────────────────────────────────

def compress_lm(
    model_id: str,
    output_dir: str | Path,
    *,
    total_budget_gb: float = 3.0,
    audio_tokenizer_gb: float = 0.81,
    max_length: int = 64,
    num_calibration_samples: int = 32,
    device: str = "cuda:0",
) -> Path:
    """Compress the LLM backbone of an OmniVoice model with AutoBit (8-bit GPTQ).

    Requires ``onecomp`` (Python >= 3.12).

    The LLM is first extracted and saved as a standalone HuggingFace model,
    then compressed with OneCompression AutoBit targeting::

        lm_budget_gb = total_budget_gb - audio_tokenizer_gb

    The compressed model is saved to ``output_dir`` and can be loaded back with
    :func:`load_compressed_model`.

    Args:
        model_id: HuggingFace model ID or local path of the OmniVoice checkpoint.
        output_dir: Directory where the compressed LLM will be saved.
        total_budget_gb: Target total weight VRAM (LM + audio tokenizer).
            Defaults to 3.0 GB.
        audio_tokenizer_gb: Expected VRAM of the audio tokenizer (FP16).
            Defaults to 0.81 GB.
        max_length: Calibration sequence length.  Keep low (≤ 64) to avoid
            OOM from the extended-vocabulary ``lm_head`` during calibration.
        num_calibration_samples: Number of calibration samples.
        device: CUDA device for compression.

    Returns:
        Path to the saved compressed model directory.
    """
    try:
        from onecomp import setup_logger, ModelConfig, Runner, AutoBitQuantizer, GPTQ
        from onecomp.utils import estimate_wbits_from_vram
    except ImportError as e:
        raise ImportError(
            "onecomp is required for LM compression. "
            "Install it with: pip install onecomp  (Python >= 3.12)"
        ) from e

    from ._bootstrap import ensure_omnivoice_on_path
    ensure_omnivoice_on_path()

    import torch, gc as _gc
    from omnivoice.models.omnivoice import OmniVoice

    setup_logger()
    output_dir = Path(output_dir)
    llm_extract_dir = output_dir.parent / (output_dir.name + "_llm_extracted")
    output_dir.mkdir(parents=True, exist_ok=True)
    llm_extract_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: extract LLM
    if not (llm_extract_dir / "model.safetensors").exists():
        print(f"Loading {model_id} to extract LLM ...")
        model = OmniVoice.from_pretrained(model_id, attn_implementation="eager", load_asr=False)
        model = model.to("cpu")
        llm = model.llm
        llm.save_pretrained(str(llm_extract_dir))
        model.text_tokenizer.save_pretrained(str(llm_extract_dir))
        del model, llm
        _gc.collect()
        print(f"LLM extracted to {llm_extract_dir}")
    else:
        print(f"LLM already extracted at {llm_extract_dir}, skipping.")

    # Step 2: estimate bitwidth
    lm_budget = total_budget_gb - audio_tokenizer_gb
    print(f"Estimating bitwidth for LM budget = {lm_budget:.2f} GB ...")
    result = estimate_wbits_from_vram(str(llm_extract_dir), total_vram_gb=lm_budget)
    target_bit = result.target_bitwidth
    print(f"Target bitwidth: {target_bit:.2f} bpw")

    # Step 3: compress
    quantizer = AutoBitQuantizer(
        assignment_strategy="activation_aware",
        target_bit=target_bit,
        quantizers=[GPTQ(wbits=b) for b in (2, 3, 4, 8)],
        save_path=str(output_dir),
    )
    runner = Runner(
        model_config=ModelConfig(path=str(llm_extract_dir), device=device),
        quantizer=quantizer,
        qep=False,
        max_length=max_length,
        num_calibration_samples=num_calibration_samples,
    )
    runner.run()
    runner.save_quantized_model(str(output_dir))

    # Write a marker so load_model can detect this is a compressed LLM dir
    marker = output_dir / "omnivoice_compressed_llm.json"
    marker.write_text(
        json.dumps({"source_model": model_id, "total_budget_gb": total_budget_gb}, indent=2),
        encoding="utf-8",
    )
    print(f"Compressed LLM saved to {output_dir}")
    return output_dir


# ── Loading compressed model ──────────────────────────────────────────────────

def is_compressed_llm_dir(checkpoint_dir: str | Path) -> bool:
    """Return True if ``checkpoint_dir`` is a OneComp-compressed LLM directory."""
    p = Path(checkpoint_dir)
    return (p / "omnivoice_compressed_llm.json").exists() or (
        (p / "config.json").exists()
        and "quantization_config" in (p / "config.json").read_text(encoding="utf-8")
        and "gptq" in (p / "config.json").read_text(encoding="utf-8").lower()
    )


def load_compressed_model(
    base_model: str,
    compressed_llm_dir: str | Path,
    device: str = "cuda:0",
    strip_encoder: bool = True,
) -> "OmniVoice":
    """Load OmniVoice with a OneComp-compressed LLM backbone.

    Args:
        base_model: HuggingFace model ID (used to load the audio tokenizer).
        compressed_llm_dir: Directory produced by :func:`compress_lm`.
        device: Target device.
        strip_encoder: If True, remove encoder-only modules from the audio
            tokenizer (~715 MB saved, safe for no-ref inference).

    Returns:
        Loaded ``OmniVoice`` model ready for inference.
    """
    try:
        from onecomp import load_quantized_model
    except ImportError as e:
        raise ImportError(
            "onecomp is required to load compressed models. "
            "Install it with: pip install onecomp  (Python >= 3.12)"
        ) from e

    from ._bootstrap import ensure_omnivoice_on_path
    ensure_omnivoice_on_path()

    from omnivoice.models.omnivoice import OmniVoice

    print(f"Loading base model (audio tokenizer) from {base_model} ...")
    model = OmniVoice.from_pretrained(base_model, attn_implementation="eager", load_asr=False)

    print(f"Loading compressed LLM from {compressed_llm_dir} ...")
    compressed_causal, _ = load_quantized_model(str(compressed_llm_dir))
    model.llm = compressed_causal.model.to(device)

    if strip_encoder:
        strip_audio_encoder(model)

    model = model.to(device)
    model.eval()
    return model
