"""
Export a HuggingFace causal-LM to ONNX for use with onnxruntime-genai.

The export is handled entirely by the ort-genai model builder — this script
adds argument handling, progress reporting, and a tokenizer_config.json fix
so the output works with OgaTokenizer out of the box.

"Uncensored" behaviour comes from the model weights you choose.  Just point
this script at an uncensored HuggingFace model ID (base model or uncensored
fine-tune) and the export is identical.  No safety layer is injected here.

Usage:
    pip install onnxruntime-genai transformers torch

    # CPU int4 (smallest, recommended for prompt enhancement)
    python export_llm_onnx.py <hf_model_id_or_local_path> --output models/my-llm-onnx

    # DirectML fp16
    python export_llm_onnx.py <hf_model_id> --output models/my-llm-onnx --precision fp16 --execution_provider dml

    # CUDA int4
    python export_llm_onnx.py <hf_model_id> --output models/my-llm-onnx --precision int4 --execution_provider cuda

Example uncensored Llama-3.2 models:
    meta-llama/Llama-3.2-3B            (base — no instruction tuning)
    unsloth/Llama-3.2-3B-Instruct      (use with a custom system prompt to skip refusals)
    Any community fine-tune with "uncensored" or "abliterated" in the name

Outputs (under --output/):
    model.onnx / model.onnx.data  — the LLM in ort-genai format
    genai_config.json             — ort-genai runtime config
    tokenizer.json                — BPE vocab/merges
    tokenizer_config.json         — patched so OgaTokenizer accepts it
"""
import argparse
import json
import os
import sys
import time


# ── Helpers ───────────────────────────────────────────────────────────────────

def check_dependencies():
    missing = []
    for pkg in ("onnxruntime_genai", "transformers", "torch"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Install with:  pip install {' '.join(missing)}")
        sys.exit(1)


def patch_tokenizer_config(output_dir: str):
    """
    ort-genai's OgaTokenizer does not support PreTrainedTokenizerFast
    (which maps internally to TokenizersBackend).  Replace it with the
    concrete class name so the tokenizer loads correctly.

    Mapping covers the model families this script is likely used with:
      PreTrainedTokenizerFast  →  LlamaTokenizer   (Llama 2/3, Mistral, …)
    Extend the map below if you export GPT-2 / Phi / Falcon / etc.
    """
    path = os.path.join(output_dir, "tokenizer_config.json")
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    replacements = {
        "PreTrainedTokenizerFast": "LlamaTokenizer",
    }

    original = cfg.get("tokenizer_class", "")
    patched  = replacements.get(original, original)

    if patched != original:
        cfg["tokenizer_class"] = patched
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"  tokenizer_class: {original!r} → {patched!r}")
    else:
        print(f"  tokenizer_class: {original!r} (no patch needed)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export a HuggingFace LLM to ort-genai ONNX format.")
    parser.add_argument(
        "model",
        help="HuggingFace model ID (e.g. meta-llama/Llama-3.2-3B) "
             "or path to a local model directory.")
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output directory for the exported model.")
    parser.add_argument(
        "--precision", "-p", default="int4",
        choices=["int4", "int8", "fp16", "fp32"],
        help="Weight precision. int4 is recommended for prompt enhancement "
             "(fast, small). Default: int4")
    parser.add_argument(
        "--execution_provider", "-e", default="cpu",
        choices=["cpu", "cuda", "dml"],
        help="Target execution provider. Default: cpu")
    parser.add_argument(
        "--cache_dir",
        help="HuggingFace cache directory for downloaded model weights.")
    args = parser.parse_args()

    check_dependencies()
    from onnxruntime_genai.models.builder import create_model  # noqa: E402

    os.makedirs(args.output, exist_ok=True)

    print(f"Model:     {args.model}")
    print(f"Output:    {args.output}")
    print(f"Precision: {args.precision}")
    print(f"Provider:  {args.execution_provider}")
    print()

    t0 = time.time()
    print("Exporting — this may take several minutes …")

    # The builder downloads (if needed), converts, quantizes, and writes
    # model.onnx + genai_config.json + tokenizer files to args.output.
    create_model(
        model_name=args.model,
        input_path="",           # empty → download from HuggingFace
        output_dir=args.output,
        precision=args.precision,
        execution_provider=args.execution_provider,
        cache_dir=args.cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
    )

    elapsed = time.time() - t0
    print(f"\nExport done in {elapsed:.0f}s")

    print("Patching tokenizer_config.json …")
    patch_tokenizer_config(args.output)

    print(f"\nDone. Model written to: {args.output}")
    print("Set the path in config.json → promptEnhancer.modelDir to use it.")


if __name__ == "__main__":
    main()