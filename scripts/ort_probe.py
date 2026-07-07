"""Shared ONNX Runtime probing primitives for the export check scripts.

Used by both `verify_model.py` (the C++-parsed import smoke test) and
`validate_dynamic_sdxl.py` (the human-facing GPU-box validator). ONLY the
genuinely-identical, side-effect-free primitives live here: the ORT->numpy dtype
map, GPU-provider discovery, session construction, and the two tiny shape-dim
resolvers. Each caller keeps its own main(), output protocol (VERIFY: vs [WARN]),
exit-code gates, and feed builders — those differ on purpose and stay put.

`onnxruntime` is imported lazily inside the functions so this module (and its
pure helpers) can be imported on a host without ORT installed.
"""
from __future__ import annotations

import numpy as np

# ORT tensor-type string -> numpy dtype. ORT has no numpy bf16, so bf16 is fed as
# fp32 (accepted). Anything unlisted resolves to fp32 via np_dtype().
ORT_TO_NP = {
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(bfloat16)": np.float32,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}

GPU_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider", "ROCMExecutionProvider")


def np_dtype(ort_type: str):
    return ORT_TO_NP.get(ort_type, np.float32)


def import_ort(preload: bool = False):
    """Import onnxruntime.

    preload=True asks ORT (>=1.19) to preload CUDA/cuDNN DLLs from installed
    nvidia-*-cuXX pip wheels, so a matching `pip install nvidia-...` is enough with
    no system CUDA toolkit (no-op on older ORT). The GPU-box validator passes True;
    verify_model runs mostly on CPU and passes False.
    """
    import onnxruntime as ort
    if preload:
        fn = getattr(ort, "preload_dlls", None)
        if fn is not None:
            try:
                fn()
            except Exception:
                pass
    return ort


def gpu_providers(preload: bool = False) -> list:
    """GPU execution providers available in this ORT build (empty on CPU-only)."""
    ort = import_ort(preload=preload)
    avail = set(ort.get_available_providers())
    return [p for p in GPU_PROVIDERS if p in avail]


def make_session(path: str, *, prefer_gpu: bool, preload: bool = False,
                 basic_opt: bool = False):
    """Construct an InferenceSession with the options both scripts share.

    Always disables the CPU memory arena (its large pre-reserved pool inflates peak
    RSS when loading multi-GB fp16 models — the exact thing that OOM-kills
    verification). The two callers differ only in the parameterized knobs:

    - prefer_gpu: GPU EPs first (then CPU fallback), else CPU only.
    - preload:    preload CUDA/cuDNN DLLs (GPU-box validator only).
    - basic_opt:  cap graph optimization at BASIC — ORT's full optimizer can
                  segfault the CPU EP on fp16 Resize graphs (the nodes these
                  exports wrap). The validator sets this; the importer does not.
    """
    ort = import_ort(preload=preload)
    so = ort.SessionOptions()
    so.log_severity_level = 3  # errors only — keep the log readable
    so.enable_cpu_mem_arena = False
    if basic_opt:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    providers = (gpu_providers(preload=preload) + ["CPUExecutionProvider"]) if prefer_gpu \
        else ["CPUExecutionProvider"]
    return ort.InferenceSession(path, so, providers=providers)


def fixed_dim(dim, fallback: int) -> int:
    """Resolve one ORT session-metadata shape dim to a concrete size.

    Static graphs report ints (honoured exactly); dynamic graphs report
    strings/None, where the caller-supplied fallback test size is substituted.
    """
    return dim if isinstance(dim, int) and dim > 0 else fallback


def dim_is_dynamic(dim) -> bool:
    """True if an ONNX proto shape dim is symbolic (dim_param) or unset (no dim_value)."""
    return bool(dim.dim_param) or not dim.HasField("dim_value")
