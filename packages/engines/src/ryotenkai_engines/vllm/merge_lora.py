#!/usr/bin/env python3
"""
LoRA Merge Script for Inference Deployment

Merges a LoRA adapter with a base model and saves the result as a unified model.

Usage:
    # Local adapter directory:
    python3 merge_lora.py \\
        --base-model Qwen/Qwen2.5-0.5B-Instruct \\
        --adapter /workspace/runs/X/adapter \\
        --output /workspace/runs/X/model \\
        --cache-dir /workspace/hf_cache \\
        --trust-remote-code

    # HuggingFace repo ID:
    python3 merge_lora.py \\
        --base-model Qwen/Qwen2.5-0.5B-Instruct \\
        --adapter username/my-lora-adapter \\
        --output /workspace/runs/X/model \\
        --cache-dir /workspace/hf_cache

Output:
    - Merged model saved to --output directory (safetensors format)
    - Tokenizer saved to --output directory
    - Prints "MERGE_SUCCESS" to stdout on success

Exit Codes:
    0: Success
    1: Error (see stderr for details)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Validate dependencies
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
except ImportError as e:
    print(f"ERROR: Missing required dependency: {e}", file=sys.stderr)
    print("Install: pip install transformers peft accelerate", file=sys.stderr)
    sys.exit(1)


def _get_hf_token() -> str | None:
    """
    Single source of truth for HuggingFace token in this project.

    We intentionally use only HF_TOKEN (no HUGGINGFACE_HUB_TOKEN).
    """
    tok = os.environ.get("HF_TOKEN")
    if tok:
        tok = tok.strip()
    return tok or None


def _hub_kwargs(token: str | None) -> dict[str, Any]:
    """Prefer modern `token=...` kwarg."""
    if not token:
        return {}
    return {"token": token}


def _hub_kwargs_legacy(token: str | None) -> dict[str, Any]:
    """Fallback for older APIs that still use `use_auth_token=...`."""
    if not token:
        return {}
    return {"use_auth_token": token}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter with base model for inference deployment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Base model ID (HuggingFace repo) or local path",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="LoRA adapter: HuggingFace repo ID (user/repo) or local directory path",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for merged model",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/workspace/hf_cache",
        help="HuggingFace cache directory (for base model download)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading model (required for some models)",
    )

    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    """Validate input paths and determine adapter type."""
    adapter_path = Path(args.adapter)

    # Check if adapter is a local path or HuggingFace repo ID
    if adapter_path.exists():
        # Local path: validate structure
        adapter_config = adapter_path / "adapter_config.json"
        if not adapter_config.exists():
            raise FileNotFoundError(
                f"adapter_config.json not found in adapter directory: {args.adapter}\n"
                "This does not appear to be a valid LoRA adapter."
            )
        print(f"✓ Adapter type: local directory ({args.adapter})", flush=True)
    else:
        # Assume HuggingFace repo ID (e.g., "user/repo-name")
        # Validation will happen during PeftModel.from_pretrained()
        if "/" not in args.adapter:
            raise ValueError(
                f"Adapter '{args.adapter}' is neither a local path nor a valid HuggingFace repo ID.\n"
                "Expected format: 'username/repo-name' or local directory path."
            )
        print(f"✓ Adapter type: HuggingFace repo ID ({args.adapter})", flush=True)

    # Ensure output and cache dirs exist
    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)


def merge_adapter(args: argparse.Namespace) -> None:
    """
    Merge LoRA adapter with base model.

    Steps:
    1. Load base model from HuggingFace or local path
    2. Load LoRA adapter using PEFT
    3. Merge adapter weights into base model
    4. Save merged model + tokenizer
    """
    token = _get_hf_token()
    if token:
        print("✓ HF_TOKEN detected (auth enabled)", flush=True)
    else:
        print("⚠️ HF_TOKEN not set (public models only)", flush=True)

    print(f"[1/5] Loading base model: {args.base_model}", flush=True)
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=args.trust_remote_code,
            cache_dir=args.cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto",  # Auto-distribute across available GPUs
            **_hub_kwargs(token),
        )
    except TypeError:
        # Backward compat: older transformers use `use_auth_token=...`
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=args.trust_remote_code,
            cache_dir=args.cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto",
            **_hub_kwargs_legacy(token),
        )
    print(f"✓ Base model loaded: {base_model.config.model_type}", flush=True)

    print(f"[2/5] Loading LoRA adapter: {args.adapter}", flush=True)
    try:
        model_with_adapter = PeftModel.from_pretrained(
            base_model,
            args.adapter,
            cache_dir=args.cache_dir,
            is_trainable=False,
            **_hub_kwargs(token),
        )
    except TypeError:
        model_with_adapter = PeftModel.from_pretrained(
            base_model,
            args.adapter,
            cache_dir=args.cache_dir,
            is_trainable=False,
            **_hub_kwargs_legacy(token),
        )
    print("✓ LoRA adapter loaded", flush=True)

    print("[3/5] Merging adapter into base model...", flush=True)
    merged_model = model_with_adapter.merge_and_unload()
    print("✓ Merge completed", flush=True)

    print(f"[4/5] Saving merged model to: {args.output}", flush=True)
    merged_model.save_pretrained(
        args.output,
        safe_serialization=True,  # Use safetensors format (recommended)
    )
    print(f"✓ Merged model saved: {args.output}/model.safetensors", flush=True)

    print(f"[5/5] Saving tokenizer to: {args.output}", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            trust_remote_code=args.trust_remote_code,
            cache_dir=args.cache_dir,
            **_hub_kwargs(token),
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            trust_remote_code=args.trust_remote_code,
            cache_dir=args.cache_dir,
            **_hub_kwargs_legacy(token),
        )
    tokenizer.save_pretrained(args.output)
    print(f"✓ Tokenizer saved: {args.output}/tokenizer.json", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("MERGE_SUCCESS", flush=True)
    print("=" * 60, flush=True)


def main() -> None:
    """Main entrypoint."""
    args = parse_args()

    try:
        print("=" * 60, flush=True)
        print("LoRA Merge Job", flush=True)
        print("=" * 60, flush=True)
        print(f"Base model:  {args.base_model}", flush=True)
        print(f"Adapter:     {args.adapter}", flush=True)
        print(f"Output:      {args.output}", flush=True)
        print(f"Cache dir:   {args.cache_dir}", flush=True)
        print(f"Trust code:  {args.trust_remote_code}", flush=True)
        print("=" * 60 + "\n", flush=True)

        validate_paths(args)
        merge_adapter(args)

        sys.exit(0)

    except FileNotFoundError as e:
        print(f"\nERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Merge failed: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
