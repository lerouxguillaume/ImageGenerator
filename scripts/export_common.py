"""
Shared utilities for ONNX model export (SD 1.5 and SDXL).
"""
import gc
import importlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch


def check_dependencies(required: list[str], optional: list[str] | None = None) -> None:
    """Verify required packages are importable; warn about missing optional ones."""
    missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]
    if missing:
        pip_names = " ".join(missing)
        raise ImportError(
            f"Missing required package(s): {', '.join(missing)}\n"
            f"Install with:  pip install {pip_names}"
        )
    for pkg in (optional or []):
        if importlib.util.find_spec(pkg) is None:
            print(f"  Optional package '{pkg}' not installed; some features will be skipped")


def check_model_file(path: str) -> None:
    """Verify the source checkpoint exists and looks reasonable."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            f"Pass the full path to a .safetensors or .ckpt checkpoint."
        )
    size_gb = os.path.getsize(path) / 1e9
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".safetensors", ".ckpt", ".pt"):
        print(f"  Warning: unexpected file extension '{ext}'")
    print(f"  Source: {path}  ({size_gb:.1f} GB)")


@contextmanager
def export_step(name: str):
    """Times a named export step and re-raises exceptions with context."""
    print(f"\n[{name}]")
    t0 = time.time()
    try:
        yield
        print(f"  Done in {time.time() - t0:.1f}s")
    except MemoryError as e:
        raise MemoryError(
            f"Step '{name}' ran out of memory.\n"
            f"Close other applications or try exporting on a machine with more RAM."
        ) from e
    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError(
            f"Step '{name}' ran out of GPU memory.\n"
            f"Try CPU-only export or reduce batch size."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Step '{name}' failed: {e}") from e


def patch_clip_for_tracing() -> None:
    """Replace create_causal_mask with a tracing-safe implementation."""
    import transformers.models.clip.modeling_clip as _clip_mod

    def _safe_causal_mask(config, inputs_embeds,
                           attention_mask=None, past_key_values=None, **_):
        seq = inputs_embeds.shape[1]
        mask = torch.full((1, 1, seq, seq), float("-inf"),
                          dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        return torch.triu(mask, diagonal=1)

    _clip_mod.create_causal_mask = _safe_causal_mask


def patch_fp32_upcasts_for_tracing() -> None:
    """Eliminate fp32-upcast patterns that produce Cast-to-float32 nodes in fp16 exports.

    Three sources in diffusers 0.37.x:

    1. get_timestep_embedding  — uses torch.arange(..., dtype=float32) and an
       explicit .float() call, so the whole sinusoidal embedding runs in fp32
       regardless of the input timestep dtype.  Fixed by replacing both with
       the input tensor's dtype.

    2. FP32SiLU — a nn.Module that does F.silu(x.float()).to(x.dtype), adding
       a redundant fp32 roundtrip in the timestep-embedding MLP.  Fixed by
       replacing its forward with a plain fp16-safe F.silu(x).

    Both patches are global monkey-patches applied before the pipeline is
    loaded, so they are in effect when the tracer runs.  They are safe for
    inference: the outputs are mathematically identical up to fp16 precision,
    which is the precision the model was already exported at.
    """
    import math as _math

    # ── 1. get_timestep_embedding ─────────────────────────────────────────────
    try:
        import diffusers.models.embeddings as _emb
        import torch.nn.functional as _F

        def _patched_gte(
            timesteps: torch.Tensor,
            embedding_dim: int,
            flip_sin_to_cos: bool = False,
            downscale_freq_shift: float = 1,
            scale: float = 1,
            max_period: int = 10000,
        ) -> torch.Tensor:
            assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
            half_dim = embedding_dim // 2
            exponent = -_math.log(max_period) * torch.arange(
                start=0, end=half_dim,
                dtype=timesteps.dtype,   # was hard-coded float32
                device=timesteps.device,
            )
            exponent = exponent / (half_dim - downscale_freq_shift)
            emb = torch.exp(exponent)
            emb = timesteps[:, None] * emb[None, :]  # was timesteps[:, None].float()
            emb = scale * emb
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
            if flip_sin_to_cos:
                emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
            if embedding_dim % 2 == 1:
                emb = _F.pad(emb, (0, 1, 0, 0))
            return emb

        _emb.get_timestep_embedding = _patched_gte
        print("  Patched get_timestep_embedding to use input dtype")
    except (ImportError, AttributeError) as e:
        print(f"  Warning: could not patch get_timestep_embedding ({e})")

    # ── 2. FP32SiLU ───────────────────────────────────────────────────────────
    try:
        import torch.nn.functional as _F
        from diffusers.models.activations import FP32SiLU as _FP32SiLU

        def _fp16_silu_forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return _F.silu(inputs)

        _FP32SiLU.forward = _fp16_silu_forward
        print("  Patched FP32SiLU to use input dtype")
    except (ImportError, AttributeError) as e:
        print(f"  Warning: could not patch FP32SiLU ({e})")


def disable_attention_upcasting(unet: torch.nn.Module) -> None:
    """Set upcast_attention=False and upcast_softmax=False on every Attention module.

    Many SDXL checkpoints ship with upcast_attention=True in their config,
    which makes every attention block cast query and key to float32 before the
    dot-product and cast back afterwards.  In a traced fp16 UNet this generates
    two Cast(fp16→float32) nodes and one Cast(float32→fp16) per attention
    operation.  Disabling the flags makes the entire attention path stay in
    fp16, consistent with the rest of the graph.
    """
    try:
        from diffusers.models.attention_processor import Attention as _Attn
    except ImportError:
        return

    count = 0
    for module in unet.modules():
        if isinstance(module, _Attn) and (module.upcast_attention or module.upcast_softmax):
            module.upcast_attention = False
            module.upcast_softmax = False
            count += 1
    if count:
        print(f"  Disabled upcast_attention/upcast_softmax on {count} Attention module(s)")


@dataclass(frozen=True)
class ExportComponentSpec:
    step_name: str
    component_name: str
    filename: str
    model: torch.nn.Module
    dummy_inputs: Any
    input_names: list[str]
    output_names: list[str]
    dynamic_axes: dict[str, dict[int, str]] | None = None
    do_constant_folding: bool = False
    exporter: str = "legacy"
    fix_fp32_constants: bool = False
    fix_attention_sqrt_cast: bool = False
    fix_resize_fp16: bool = False
    simplify: bool = False
    force_external_data: bool = False
    release_after: tuple[Any, ...] = ()


class ExportPolicy:
    """Shared export policy surface with model-specific overrides."""

    model_type = "base"

    def text_encoder_dynamic_axes(
        self,
        input_name: str,
        output_names: list[str],
    ) -> dict[str, dict[int, str]]:
        axes = {input_name: {0: "batch"}}
        for name in output_names:
            axes[name] = {0: "batch"}
        return axes

    def unet_dynamic_axes(self) -> dict[str, dict[int, str]]:
        raise NotImplementedError

    def vae_dynamic_axes(self) -> dict[str, dict[int, str]] | None:
        raise NotImplementedError

    def unet_exporter(self) -> str:
        return "legacy"

    def vae_exporter(self) -> str:
        return "legacy"

    def should_fix_fp32_constants(self, component_name: str) -> bool:
        return False

    def should_fix_attention_sqrt_cast(self, component_name: str) -> bool:
        return False

    def should_fix_resize_fp16(self, component_name: str) -> bool:
        return False

    def should_simplify_vae(self, requested: bool) -> bool:
        return requested

    def should_force_external_data(self, component_name: str) -> bool:
        return False


class SD15ExportPolicy(ExportPolicy):
    model_type = "sd15"

    def unet_dynamic_axes(self) -> dict[str, dict[int, str]]:
        return {
            "latent": {0: "batch", 2: "height", 3: "width"},
            "timestep": {0: "batch"},
            "encoder_hidden_states": {0: "batch"},
            "latent_out": {0: "batch", 2: "height", 3: "width"},
        }

    def vae_dynamic_axes(self) -> dict[str, dict[int, str]]:
        return {
            "latent": {0: "batch", 2: "height", 3: "width"},
            "image": {0: "batch", 2: "height", 3: "width"},
        }


class SDXLExportPolicy(ExportPolicy):
    model_type = "sdxl"

    def unet_dynamic_axes(self) -> dict[str, dict[int, str]]:
        return {
            "latent": {0: "batch"},
            "timestep": {0: "batch"},
            "encoder_hidden_states": {0: "batch"},
            "text_embeds": {0: "batch"},
            "time_ids": {0: "batch"},
            "latent_out": {0: "batch"},
        }

    def vae_dynamic_axes(self) -> None:
        return None

    def vae_exporter(self) -> str:
        return "dynamo"

    def should_fix_fp32_constants(self, component_name: str) -> bool:
        return component_name in {"unet", "vae_decoder"}

    def should_fix_attention_sqrt_cast(self, component_name: str) -> bool:
        return component_name == "unet"

    def should_fix_resize_fp16(self, component_name: str) -> bool:
        return component_name == "unet"

    def should_simplify_vae(self, requested: bool) -> bool:
        return requested

    def should_force_external_data(self, component_name: str) -> bool:
        return component_name in {
            "text_encoder",
            "text_encoder_2",
            "unet",
            "vae_decoder",
        }


def _normalize_dummy_inputs(dummy_inputs: Any) -> tuple[Any, ...]:
    if isinstance(dummy_inputs, tuple):
        return dummy_inputs
    return (dummy_inputs,)


def _supports_batch_only_dynamic_shapes(
    input_names: list[str],
    dynamic_axes: dict[str, dict[int, str]] | None,
) -> bool:
    if not dynamic_axes or not hasattr(torch, "export"):
        return False
    for input_name in input_names:
        axes = dynamic_axes.get(input_name, {})
        if any(axis != 0 for axis in axes):
            return False
    return True


def _build_dynamic_shapes(
    input_names: list[str],
    dynamic_axes: dict[str, dict[int, str]] | None,
) -> dict[str, dict[int, Any]] | None:
    if not _supports_batch_only_dynamic_shapes(input_names, dynamic_axes):
        return None
    batch_dim = torch.export.Dim("batch")
    dynamic_shapes: dict[str, dict[int, Any]] = {}
    for input_name in input_names:
        axes = dynamic_axes.get(input_name, {}) if dynamic_axes else {}
        if axes:
            dynamic_shapes[input_name] = {0: batch_dim}
    return dynamic_shapes or None


def _legacy_export(
    model: torch.nn.Module,
    dummy_inputs: tuple[Any, ...],
    path: str,
    *,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]] | None,
    do_constant_folding: bool,
) -> None:
    # The legacy tracer path preserves the initializer layout and names the LoRA
    # patcher expects for SD 1.5 and SDXL shared components.
    torch.onnx.export(
        model,
        dummy_inputs,
        path,
        opset_version=18,
        dynamo=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes or {},
        do_constant_folding=do_constant_folding,
        keep_initializers_as_inputs=True,
        export_params=True,
    )


def _dynamo_export(
    model: torch.nn.Module,
    dummy_inputs: tuple[Any, ...],
    path: str,
    *,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]] | None,
    do_constant_folding: bool,
) -> None:
    export_kwargs = {
        "opset_version": 18,
        "dynamo": True,
        "input_names": input_names,
        "output_names": output_names,
        "export_params": True,
        "do_constant_folding": do_constant_folding,
        "external_data": True,
    }
    dynamic_shapes = _build_dynamic_shapes(input_names, dynamic_axes)
    if dynamic_shapes is not None:
        export_kwargs["dynamic_shapes"] = dynamic_shapes

    try:
        torch.onnx.export(
            model,
            dummy_inputs,
            path,
            **export_kwargs,
        )
    except TypeError:
        export_kwargs.pop("do_constant_folding", None)
        torch.onnx.export(
            model,
            dummy_inputs,
            path,
            **export_kwargs,
        )


def onnx_export(
    model: torch.nn.Module,
    dummy_inputs: Any,
    path: str,
    *,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    do_constant_folding: bool = False,
    exporter: str = "legacy",
) -> None:
    """Project-wide ONNX export wrapper with selective modern-export support."""
    name = os.path.basename(path)
    dummy_inputs = _normalize_dummy_inputs(dummy_inputs)
    print(f"  Exporting {name} ...")

    if exporter not in {"legacy", "dynamo", "auto"}:
        raise ValueError(f"Unknown exporter '{exporter}'")

    if exporter in {"dynamo", "auto"}:
        can_use_dynamo = dynamic_axes is None or _supports_batch_only_dynamic_shapes(
            input_names,
            dynamic_axes,
        )
        if can_use_dynamo:
            try:
                _dynamo_export(
                    model,
                    dummy_inputs,
                    path,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    do_constant_folding=do_constant_folding,
                )
                size_mb = os.path.getsize(path) / 1e6
                print(f"  Saved {name}  ({size_mb:.0f} MB) via dynamo exporter")
                return
            except Exception as e:
                if exporter == "dynamo":
                    raise
                print(f"  Dynamo export failed ({e}); falling back to legacy tracer")
        elif exporter == "dynamo":
            raise ValueError(
                "Dynamo exporter only supports fully static graphs or batch-only dynamic axes "
                "in this project"
            )

    _legacy_export(
        model,
        dummy_inputs,
        path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=do_constant_folding,
    )
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Saved {name}  ({size_mb:.0f} MB)")


def _convert_tensor_proto_fp16(tensor_proto) -> bool:
    import numpy as np
    from onnx import TensorProto, numpy_helper

    if tensor_proto.data_type != TensorProto.FLOAT:
        return False
    if tensor_proto.data_location == TensorProto.EXTERNAL:
        return False
    array = numpy_helper.to_array(tensor_proto).astype(np.float16)
    tensor_proto.CopyFrom(numpy_helper.from_array(array, tensor_proto.name))
    return True


def fix_fp32_constants(path: str) -> None:
    """Convert embedded fp32 constants to fp16 without reloading external weights.

    Handles three forms the ONNX tracer can emit for Constant nodes:
    - attr.t (TensorProto)    — multi-element tensor constant
    - value_float (scalar)    — e.g. attention scale 1/sqrt(d_k), traced as attr.f
    - value_floats (list)     — fp32 list constant, traced as attr.floats

    The scalar/list forms cannot be modified in-place because their proto field
    type (FLOAT/FLOATS) differs from the replacement type (TENSOR).  They are
    rebuilt as zero-dim / 1-D fp16 TensorProto "value" attributes and the old
    attribute is removed from the node.
    """
    import numpy as np
    import onnx
    from onnx import AttributeProto, numpy_helper

    model = onnx.load_model(path, load_external_data=False)
    changed = 0

    for initializer in model.graph.initializer:
        if _convert_tensor_proto_fp16(initializer):
            changed += 1

    for node in model.graph.node:
        if node.op_type != "Constant":
            continue

        replacements: dict[int, AttributeProto] = {}
        for i, attr in enumerate(node.attribute):
            if attr.HasField("t"):
                if _convert_tensor_proto_fp16(attr.t):
                    changed += 1
            elif attr.type == AttributeProto.FLOAT:
                # Scalar value_float → 0-d fp16 TensorProto stored as "value"
                tp = numpy_helper.from_array(np.array(attr.f, dtype=np.float16))
                replacements[i] = onnx.helper.make_attribute("value", tp)
                changed += 1
            elif attr.type == AttributeProto.FLOATS:
                # value_floats list → 1-D fp16 TensorProto stored as "value"
                arr = np.array(list(attr.floats), dtype=np.float16)
                tp = numpy_helper.from_array(arr)
                replacements[i] = onnx.helper.make_attribute("value", tp)
                changed += 1

        if replacements:
            new_attrs = [
                replacements[i] if i in replacements else attr
                for i, attr in enumerate(node.attribute)
            ]
            del node.attribute[:]
            node.attribute.extend(new_attrs)

    if changed == 0:
        return

    data_location = os.path.basename(path) + ".data"
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        location=data_location,
        all_tensors_to_one_file=True,
        convert_attribute=False,
    )
    print(f"  Fixed {changed} embedded fp32 constant(s) in-process")


def fix_attention_sqrt_cast_fp32(path: str) -> None:
    """Rewrite Cast(to=float32) nodes whose input is a Sqrt to Cast(to=float16).

    Some diffusers attention implementations compute the attention scale as
    `sqrt(head_dim).to(float32)`, producing a Sqrt → Cast(fp32) chain.  In an
    fp16 UNet this causes ORT type errors (tensor(float16) vs tensor(float)) in
    the downstream Div or Mul that consumes the scale.

    The fix is very narrow: only Cast nodes whose *direct* input comes from a
    Sqrt node are rewritten.  This pattern never appears in Resize scale paths
    (which use literal Constant values, not computed Sqrt values), so it is safe
    to rewrite unconditionally.
    """
    import onnx
    from onnx import TensorProto

    model = onnx.load_model(path, load_external_data=False)

    out_to_node: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for out in node.output:
            out_to_node[out] = node

    changed = 0
    for node in model.graph.node:
        if node.op_type != "Cast" or not node.input:
            continue
        parent = out_to_node.get(node.input[0])
        if parent is None or parent.op_type != "Sqrt":
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i == TensorProto.FLOAT:
                attr.i = TensorProto.FLOAT16
                changed += 1

    if changed == 0:
        return

    data_location = os.path.basename(path) + ".data"
    onnx.save_model(
        model, path,
        save_as_external_data=True,
        location=data_location,
        all_tensors_to_one_file=True,
        convert_attribute=False,
    )
    print(f"  Fixed {changed} Cast(Sqrt→float32) → Cast(Sqrt→float16)")


def fix_resize_fp16_input(path: str) -> None:
    """Wrap Resize data inputs with Cast(fp16→fp32) / Cast(fp32→fp16) pairs.

    ORT rejects fp16 as the data type for the Resize operator on some builds
    and execution providers.  In an fp16 UNet every Resize data input will be
    fp16.  This function inserts explicit Cast nodes so the Resize op always
    receives and emits fp32 data, keeping the rest of the graph in fp16.

    Only Resize nodes whose data input (input[0]) is inferred to be fp16 are
    patched.  Resize ops that already receive fp32 (e.g. a correctly-typed
    static model) are left untouched.
    """
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load_model(path, load_external_data=False)

    # Build a set of Cast(to=fp32) outputs already wrapping a Resize data input
    # so we can skip nodes that were patched by a previous run.
    out_to_node: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for out in node.output:
            out_to_node[out] = node

    to_patch: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.op_type != "Resize" or not node.input:
            continue
        parent = out_to_node.get(node.input[0])
        if parent is not None and parent.op_type == "Cast":
            for attr in parent.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT:
                    break  # already wrapped — skip
            else:
                to_patch.append(node)
        else:
            to_patch.append(node)

    if not to_patch:
        return

    patched_ids = {id(n) for n in to_patch}
    final_nodes: list[onnx.NodeProto] = []
    counter = 0

    for node in model.graph.node:
        if id(node) not in patched_ids:
            final_nodes.append(node)
            continue

        data_in  = node.input[0]
        data_out = node.output[0]
        suffix   = f"_resize{counter}"
        fp32_in  = data_in  + suffix + "_f32"
        fp32_out = data_out + suffix + "_f32"
        counter += 1

        final_nodes.append(helper.make_node(
            "Cast", inputs=[data_in], outputs=[fp32_in],
            to=TensorProto.FLOAT, name=node.name + "/cast_in",
        ))
        node.input[0]  = fp32_in
        node.output[0] = fp32_out
        final_nodes.append(node)
        final_nodes.append(helper.make_node(
            "Cast", inputs=[fp32_out], outputs=[data_out],
            to=TensorProto.FLOAT16, name=node.name + "/cast_out",
        ))

    del model.graph.node[:]
    model.graph.node.extend(final_nodes)

    data_location = os.path.basename(path) + ".data"
    onnx.save_model(
        model, path,
        save_as_external_data=True,
        location=data_location,
        all_tensors_to_one_file=True,
        convert_attribute=False,
    )
    print(f"  Wrapped {len(to_patch)} Resize node(s) with Cast(fp16↔fp32) pairs")


def ensure_external_data(path: str) -> None:
    """Rewrite a model into <name>.onnx + <name>.onnx.data layout.

    Also deletes any old per-tensor sidecar files that the model previously
    referenced, so the output directory is left with exactly two files:
    <name>.onnx and <name>.onnx.data.
    """
    import onnx
    from onnx import external_data_helper

    # Use absolute path so relative tensor file references resolve correctly
    # regardless of the working directory the script was launched from.
    abs_path = os.path.abspath(path)
    base_dir = os.path.dirname(abs_path)
    data_location = os.path.basename(abs_path) + ".data"

    # First pass: record which files the model currently references so we can
    # delete them precisely after consolidation (no prefix heuristics needed).
    model = onnx.load_model(abs_path, load_external_data=False)
    old_ext_files: set[str] = set()
    for init in model.graph.initializer:
        if init.data_location == 1:  # TensorProto.EXTERNAL
            for kv in init.external_data:
                if kv.key == "location":
                    old_ext_files.add(kv.value)

    # Second pass: load all tensor data into memory, inline it, then
    # re-externalize everything into a single consolidated file.
    external_data_helper.load_external_data_for_model(model, base_dir)
    external_data_helper.convert_model_from_external_data(model)

    data_path = os.path.join(base_dir, data_location)
    if os.path.exists(data_path):
        os.remove(data_path)

    external_data_helper.convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=data_location,
        size_threshold=0,
        convert_attribute=False,
    )
    onnx.save(model, abs_path)

    # Delete the old external files (individual per-tensor sidecars or a
    # previous .data file with a different name) that are no longer referenced.
    removed = 0
    for fname in old_ext_files:
        if fname == data_location:
            continue  # new consolidated file — keep it
        fpath = os.path.join(base_dir, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)
            removed += 1
    if removed:
        print(f"  Removed {removed} old external tensor file(s)")
    print(f"  Enforced external data layout: {os.path.basename(abs_path)} + {data_location}")


def cleanup_stale_component_files(path: str) -> None:
    """Remove legacy per-tensor sidecars left by previous exports for one component."""
    directory = os.path.dirname(path)
    basename = os.path.basename(path)
    stem, _ = os.path.splitext(basename)
    keep = {basename, f"{basename}.data"}
    prefix = f"{stem}."

    removed = 0
    for entry in os.listdir(directory):
        if entry in keep:
            continue
        if not entry.startswith(prefix):
            continue
        full_path = os.path.join(directory, entry)
        if not os.path.isfile(full_path):
            continue
        os.remove(full_path)
        removed += 1

    if removed:
        print(f"  Removed {removed} stale sidecar file(s) for {basename}")


def simplify_with_onnxsim(path: str) -> None:
    """Run onnxsim to fold remaining dynamic shape ops."""
    try:
        import onnx
        import onnxsim

        model = onnx.load(path)
        model_sim, ok = onnxsim.simplify(model)
        if ok:
            onnx.save(model_sim, path)
            print(f"  Simplified {os.path.basename(path)} with onnxsim")
        else:
            print("  Warning: onnxsim check failed; keeping original")
    except ImportError:
        pass
    except Exception as e:
        print(f"  Warning: onnxsim failed ({e}); keeping original")


def write_model_json(output_dir: str, model_type: str) -> None:
    """Write model.json consumed by the C++ runtime to detect the model family."""
    path = os.path.join(output_dir, "model.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"type": model_type}, f)
    print(f"  Saved model.json  (type={model_type})")


def export_component_to_dir(output_dir: str, spec: ExportComponentSpec) -> None:
    with export_step(spec.step_name):
        path = os.path.join(output_dir, spec.filename)
        cleanup_stale_component_files(path)
        onnx_export(
            spec.model,
            spec.dummy_inputs,
            path,
            input_names=spec.input_names,
            output_names=spec.output_names,
            dynamic_axes=spec.dynamic_axes,
            do_constant_folding=spec.do_constant_folding,
            exporter=spec.exporter,
        )
        if spec.fix_fp32_constants:
            fix_fp32_constants(path)
        if spec.fix_attention_sqrt_cast:
            fix_attention_sqrt_cast_fp32(path)
        if spec.fix_resize_fp16:
            fix_resize_fp16_input(path)
        if spec.simplify:
            simplify_with_onnxsim(path)
        if spec.force_external_data:
            ensure_external_data(path)
            cleanup_stale_component_files(path)
        if spec.release_after:
            free(*spec.release_after)


def free(*objs) -> None:
    """Delete objects and run gc.collect() to release memory between exports."""
    for obj in objs:
        del obj
    gc.collect()


class VAEDecoderWrapper(torch.nn.Module):
    """Wrap diffusers VAE decoder and apply latent unscale internally."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        return self.vae.decode(latent / self.vae.config.scaling_factor).sample
