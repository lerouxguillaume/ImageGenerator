"""
Shared utilities for ONNX model export (SD 1.5 and SDXL).
"""
from __future__ import annotations

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
    export_lora_weights: bool = False  # if True, saves <stem>_weights.safetensors
    skip_if_complete: bool = False     # skip export if all output files already exist
    validate: bool = False             # run ORT forward pass after export to catch runtime errors
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


class SD15ExportPolicy(ExportPolicy):
    model_type = "sd15"

    def unet_dynamic_axes(self) -> dict[str, dict[int, str]]:
        return {
            "latent": {0: "batch", 2: "height", 3: "width"},
            "timestep": {0: "batch"},
            "encoder_hidden_states": {0: "batch"},
            "latent_out": {0: "batch", 2: "height", 3: "width"},
        }

    def vae_dynamic_axes(self) -> None:
        # Static shape (1, 4, 64, 64) → (1, 3, 512, 512).
        # Dynamic H/W caused torch.onnx.export to hang for 1+ hour due to the
        # shape-propagation overhead of the many upsampling ops in the legacy
        # tracer.  SD 1.5 is fixed at 512×512 in the C++ config anyway.
        return None

    def vae_exporter(self) -> str:
        # Dynamo writes external_data=True during export, bypassing the slow
        # inline-protobuf consolidation round-trip that legacy requires.
        return "dynamo"


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
        return True  # VAE decoder also has fp16 Resize nodes at runtime

    def should_simplify_vae(self, requested: bool) -> bool:
        return requested


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
    """Fix fp16 inputs on Resize nodes so ORT accepts the model.

    ORT rejects fp16 for both:
    - input[0] (data)  — wraps with Cast(fp16→fp32) before + Cast(fp32→fp16) after
    - input[2] (scales) — wraps with Cast(fp16→fp32) only; scales must be fp32 and
                          are not consumed downstream so no fp16 restore is needed.

    In a fully-fp16 SDXL UNet, dynamic scale tensors computed via Concat/Shape ops
    come out as fp16, triggering ORT's "Type 'tensor(float16)' of input parameter
    … of operator (Resize) is invalid" error.

    Already-patched nodes (producer is already Cast(to=FLOAT)) are skipped so the
    function is safe to call multiple times.
    """
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load_model(path, load_external_data=False)

    out_to_node: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for out in node.output:
            out_to_node[out] = node

    def _already_cast_fp32(tensor_name: str) -> bool:
        """Return True if tensor_name is already produced by a Cast(to=FLOAT)."""
        producer = out_to_node.get(tensor_name)
        if producer is None or producer.op_type != "Cast":
            return False
        return any(a.name == "to" and a.i == TensorProto.FLOAT for a in producer.attribute)

    # Collect Resize nodes whose data input (input[0]) needs wrapping.
    data_to_patch: list[onnx.NodeProto] = [
        n for n in model.graph.node
        if n.op_type == "Resize" and n.input and not _already_cast_fp32(n.input[0])
    ]

    # Collect Resize nodes whose scales input (input[2]) needs a fp16→fp32 cast.
    # Only patch when input[2] is non-empty (scales mode) and not already fp32.
    scales_to_patch: list[onnx.NodeProto] = [
        n for n in model.graph.node
        if n.op_type == "Resize"
        and len(n.input) > 2 and n.input[2]
        and not _already_cast_fp32(n.input[2])
    ]

    if not data_to_patch and not scales_to_patch:
        return

    data_ids   = {id(n) for n in data_to_patch}
    scales_ids = {id(n) for n in scales_to_patch}

    final_nodes: list[onnx.NodeProto] = []
    data_counter   = 0
    scales_counter = 0

    for node in model.graph.node:
        need_data   = id(node) in data_ids
        need_scales = id(node) in scales_ids

        if not need_data and not need_scales:
            final_nodes.append(node)
            continue

        if need_data:
            data_in  = node.input[0]
            data_out = node.output[0]
            fp32_in  = data_in  + f"_resize{data_counter}_f32"
            fp32_out = data_out + f"_resize{data_counter}_f32"
            data_counter += 1
            final_nodes.append(helper.make_node(
                "Cast", inputs=[data_in], outputs=[fp32_in],
                to=TensorProto.FLOAT, name=node.name + "/cast_in",
            ))
            node.input[0]  = fp32_in
            node.output[0] = fp32_out

        if need_scales:
            scales_in = node.input[2]
            scales_f32 = scales_in + f"_scales{scales_counter}_f32"
            scales_counter += 1
            final_nodes.append(helper.make_node(
                "Cast", inputs=[scales_in], outputs=[scales_f32],
                to=TensorProto.FLOAT, name=node.name + "/cast_scales",
            ))
            node.input[2] = scales_f32

        final_nodes.append(node)

        if need_data:
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
    if data_to_patch:
        print(f"  Wrapped {len(data_to_patch)} Resize data input(s) with Cast(fp16↔fp32) pairs")
    if scales_to_patch:
        print(f"  Wrapped {len(scales_to_patch)} Resize scales input(s) with Cast(fp16→fp32)")


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
    """Run onnxsim to fold remaining dynamic shape ops.

    Preserves the .onnx.data layout if the model was already consolidated.
    """
    try:
        import onnx
        import onnxsim

        model = onnx.load(path)  # loads all data so onnxsim can run inference
        model_sim, ok = onnxsim.simplify(model)
        if ok:
            data_path = path + ".data"
            if os.path.exists(data_path):
                # Re-save with external data so we don't produce an inline-only
                # file that may exceed the protobuf 2 GB limit.
                onnx.save_model(
                    model_sim,
                    path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=os.path.basename(data_path),
                    size_threshold=0,
                )
            else:
                onnx.save(model_sim, path)
            print(f"  Simplified {os.path.basename(path)} with onnxsim")
        else:
            print("  Warning: onnxsim check failed; keeping original")
    except ImportError:
        pass
    except Exception as e:
        print(f"  Warning: onnxsim failed ({e}); keeping original")


def write_model_json(output_dir: str, model_type: str,
                     specs: list[ExportComponentSpec] | None = None) -> None:
    """Write model.json consumed by the C++ runtime."""
    data: dict = {"type": model_type}
    if specs:
        components = {}
        for spec in specs:
            entry: dict = {"onnx": spec.filename}
            if spec.export_lora_weights:
                stem = os.path.splitext(spec.filename)[0]
                entry["weights"] = f"{stem}_weights.safetensors"
            components[spec.component_name] = entry
        data["components"] = components
    path = os.path.join(output_dir, "model.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved model.json  (type={model_type}, {len(data.get('components', {}))} component(s))")


def consolidate_external_data(path: str) -> None:
    """Consolidate per-tensor sidecar files into a single <name>.onnx.data file.

    The legacy torch.onnx.export tracer writes one sidecar file per tensor when
    the model exceeds the protobuf 2 GB limit.  This function merges them all
    into a single .onnx.data file by copying raw bytes — no tensor
    deserialisation, no numpy — so it runs in seconds rather than minutes.

    Idempotent: if the model is already in <name>.onnx + <name>.onnx.data
    layout, or has no external data at all, this is a no-op.
    """
    import onnx
    from onnx import TensorProto

    abs_path = os.path.abspath(path)
    base_dir = os.path.dirname(abs_path)
    data_location = os.path.basename(abs_path) + ".data"
    data_full_path = os.path.join(base_dir, data_location)

    model = onnx.load_model(abs_path, load_external_data=False)

    # Collect all external initializers and their current file locations
    external: list[tuple] = []
    for init in model.graph.initializer:
        if init.data_location == TensorProto.EXTERNAL:
            info = {kv.key: kv.value for kv in init.external_data}
            if "location" in info:
                external.append((init, info))

    if not external:
        # Model is fully inline — no external data at all.  Force it into the
        # <name>.onnx + <name>.onnx.data layout for consistency.  Models that
        # fit inline are small (< 2 GB protobuf limit), so the full load+resave
        # here is fast (a few seconds) compared with the large-model path above.
        model_full = onnx.load(abs_path)  # loads tensor data for inline model
        onnx.save_model(
            model_full,
            abs_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_location,
            size_threshold=0,
        )
        print(f"  Forced external layout: {os.path.basename(abs_path)} + {data_location}")
        return

    locations = {info["location"] for _, info in external}
    if locations == {data_location}:
        return  # already in the target single-file layout

    print(f"  Consolidating {len(locations)} external file(s) -> {data_location} ...")

    # Write the consolidated .data file and update tensor references in-place.
    # Use a temp file so a crash mid-write doesn't leave a half-written .data.
    tmp = data_full_path + ".tmp"
    try:
        with open(tmp, "wb") as out_f:
            for init, info in external:
                src_path = os.path.join(base_dir, info["location"])
                offset = int(info.get("offset", 0))
                length_str = info.get("length")
                length = int(length_str) if length_str is not None else -1

                with open(src_path, "rb") as src_f:
                    if offset:
                        src_f.seek(offset)
                    data = src_f.read(length if length >= 0 else None)

                # Align to 8 bytes (onnx convention)
                cur = out_f.tell()
                rem = cur % 8
                if rem:
                    out_f.write(b"\x00" * (8 - rem))

                new_offset = out_f.tell()
                out_f.write(data)

                # Update the tensor proto's external_data references
                del init.external_data[:]
                for k, v in [
                    ("location", data_location),
                    ("offset", str(new_offset)),
                    ("length", str(len(data))),
                ]:
                    kv = init.external_data.add()
                    kv.key = k
                    kv.value = v

        if os.path.exists(data_full_path):
            os.remove(data_full_path)
        os.rename(tmp, data_full_path)

    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    # Serialize the updated graph structure (no tensor bytes — those stay in
    # the .data file we just wrote).
    with open(abs_path, "wb") as f:
        f.write(model.SerializeToString())

    # Remove old per-tensor sidecar files
    removed = 0
    for loc in locations:
        if loc == data_location:
            continue
        fpath = os.path.join(base_dir, loc)
        if os.path.isfile(fpath):
            os.remove(fpath)
            removed += 1
    if removed:
        print(f"  Removed {removed} old sidecar file(s)")


def verify_export(
    path: str,
    *,
    expected_inputs: list[str],
    expected_outputs: list[str],
) -> None:
    """Verify a completed export is usable by the C++ ORT pipeline.

    Checks (without loading tensor data):
    - <name>.onnx exists
    - <name>.onnx.data exists (guaranteed by consolidate_external_data)
    - Named initializers are present (required for LoRA patching)
    - Expected input and output names are all present in the graph

    Raises RuntimeError on any failure so the export aborts early rather than
    producing a silently broken model that only crashes in C++.
    """
    import onnx

    data_path = path + ".data"
    name = os.path.basename(path)

    if not os.path.exists(path):
        raise RuntimeError(f"verify: {name} was not created")
    if not os.path.exists(data_path):
        raise RuntimeError(
            f"verify: {name}.data was not created — external data consolidation failed"
        )

    # Load graph structure only (no tensor bytes) for all remaining checks
    model = onnx.load_model(path, load_external_data=False)

    n_init = len(model.graph.initializer)
    if n_init == 0:
        raise RuntimeError(
            f"verify: {name} has 0 named initializers.\n"
            f"The model was likely exported without keep_initializers_as_inputs=True "
            f"or with do_constant_folding=True, which collapses weights into anonymous "
            f"Constant nodes.  LoRA patching will not work.  Re-export with the project "
            f"scripts."
        )

    graph_inputs  = {i.name for i in model.graph.input}
    graph_outputs = {o.name for o in model.graph.output}

    missing_in  = [n for n in expected_inputs  if n not in graph_inputs]
    missing_out = [n for n in expected_outputs if n not in graph_outputs]
    if missing_in or missing_out:
        raise RuntimeError(
            f"verify: {name} has unexpected graph interface.\n"
            + (f"  Missing inputs:  {missing_in}\n"  if missing_in  else "")
            + (f"  Missing outputs: {missing_out}\n" if missing_out else "")
            + f"  Actual inputs:   {sorted(graph_inputs)}\n"
            + f"  Actual outputs:  {sorted(graph_outputs)}"
        )

    print(f"  Verified {name}: {n_init} initializers, "
          f"inputs={sorted(graph_inputs)}, outputs={sorted(graph_outputs)}")


def _export_lora_weights(model: torch.nn.Module, output_path: str) -> None:
    """Save 2-D (linear-layer) parameters as safetensors for use as LoRA base weights.

    Keys are the PyTorch parameter paths (e.g. ``unet.down_blocks.0...weight``),
    which match the ONNX initializer names produced by the legacy tracer when
    ``keep_initializers_as_inputs=True``.  The C++ ``LoraInjector`` uses an exact
    lookup by ONNX name to retrieve these weights at inference time.

    Only rank-2 tensors are saved: LoRA adapters target linear-projection and
    MLP weights, all of which are 2-D.  Embedding tables (also 2-D) are included
    so that exotic LoRAs that target them still work.

    Tensors are written in their current dtype (fp16 for UNet/VAE, fp32 for text
    encoders), keeping the file as compact as possible.
    """
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("  Warning: safetensors not installed — skipping LoRA base-weight export")
        return

    weights = {
        name: param.data.contiguous().cpu()
        for name, param in model.named_parameters()
        if param.ndim == 2
    }

    if not weights:
        print("  Warning: no 2-D parameters found — skipping LoRA base-weight export")
        return

    save_file(weights, output_path)
    size_mb = sum(t.nbytes for t in weights.values()) / 1e6
    print(f"  Saved {len(weights)} LoRA base weights ({size_mb:.0f} MB)"
          f" → {os.path.basename(output_path)}")


def _is_component_complete(path: str, spec: ExportComponentSpec) -> bool:
    """Return True if all output files for a component already exist on disk."""
    if not os.path.exists(path):
        return False
    if not os.path.exists(path + ".data"):
        return False
    if spec.export_lora_weights:
        weights = os.path.splitext(path)[0] + "_weights.safetensors"
        if not os.path.exists(weights):
            return False
    return True


def validate_with_ort(
    path: str,
    dummy_inputs: Any,
    input_names: list[str],
) -> None:
    """Run a forward pass through the exported model using ORT on CPU.

    Loads the session, feeds the same dummy inputs used for export, and
    checks that the model runs without errors.  Catches dtype mismatches,
    missing external data, and shape errors before the C++ pipeline does.

    Warning: for large models (SDXL UNet) this loads all weights into RAM
    and can take several minutes on first run.
    """
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("  Warning: onnxruntime Python package not installed — skipping ORT validation")
        return

    print("  ORT validation: creating session ...")
    t0 = time.time()
    try:
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3  # suppress INFO/WARNING spam
        sess = ort.InferenceSession(
            path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
    except Exception as e:
        raise RuntimeError(f"ORT validation: session creation failed — {e}") from e

    inputs = _normalize_dummy_inputs(dummy_inputs)
    feed: dict[str, Any] = {}
    for name, tensor in zip(input_names, inputs):
        if isinstance(tensor, torch.Tensor):
            feed[name] = tensor.detach().cpu().numpy()
        else:
            import numpy as np
            feed[name] = np.array(tensor)

    print("  ORT validation: running forward pass ...")
    try:
        outputs = sess.run(None, feed)
    except Exception as e:
        raise RuntimeError(f"ORT validation: forward pass failed — {e}") from e

    print(f"  ORT validation passed in {time.time() - t0:.1f}s  "
          f"({len(outputs)} output(s))")


def export_component_to_dir(output_dir: str, spec: ExportComponentSpec) -> None:
    with export_step(spec.step_name):
        path = os.path.join(output_dir, spec.filename)

        # Resume: skip re-exporting components that are already complete on disk.
        if spec.skip_if_complete and _is_component_complete(path, spec):
            print(f"  Already complete — skipping")
            if spec.release_after:
                free(*spec.release_after)
            return

        cleanup_stale_component_files(path)

        # 1. Export
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

        # 2. Consolidate per-tensor sidecar files into a single .onnx.data file.
        #    Raw byte copy — no tensor deserialisation.  No-op for dynamo-exported
        #    models and for models small enough to have no external data.
        consolidate_external_data(path)

        # 3. Optional graph fixes (ONLY if explicitly enabled).
        #    Each pass loads with load_external_data=False and saves only the
        #    graph structure, leaving the .onnx.data file untouched.
        if spec.fix_fp32_constants:
            fix_fp32_constants(path)

        if spec.fix_attention_sqrt_cast:
            fix_attention_sqrt_cast_fp32(path)

        if spec.fix_resize_fp16:
            fix_resize_fp16_input(path)

        # 4. Optional simplify (LAST)
        # if spec.simplify:
        #     simplify_with_onnxsim(path)

        cleanup_stale_component_files(path)

        # 5. Verify: file layout, named initializers, and graph interface.
        #    Fails loudly here rather than silently in the C++ ORT session.
        verify_export(
            path,
            expected_inputs=spec.input_names,
            expected_outputs=spec.output_names,
        )

        # 6. Optional ORT validation: full session load + forward pass on CPU.
        #    Catches dtype errors, EP-specific issues, and output shape regressions.
        #    Slow for large models — enable with --validate.
        if spec.validate:
            validate_with_ort(path, spec.dummy_inputs, spec.input_names)

        if spec.export_lora_weights:
            weights_path = os.path.splitext(path)[0] + "_weights.safetensors"
            _export_lora_weights(spec.model, weights_path)

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
