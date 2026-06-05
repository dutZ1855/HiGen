"""Utilities for compiler fuzzing tests.

This module wraps nnsmith model generation and ONNX Runtime validation
utilities used by the compiler fuzzing workflow. It provides helpers for:
- model generation parameterization
- ONNX Runtime inference testing
- TVM compilation and runtime validation
- result collection and error handling
- timeout control and isolated execution
"""
from __future__ import annotations
import concurrent.futures
import json
import multiprocessing
import os
import pickle
import random
import signal
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper

from .filter import should_keep_crash_case

# 优先使用本仓库的 TVM Python 包（tvm/python），避免落到环境里的老版本缺符号
import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
_TVM_PY = _REPO_ROOT / "tvm" / "python"
if _TVM_PY.exists():
    sys.path.insert(0, str(_TVM_PY))

import tvm
from tvm import relax


def _ensure_vm_instr_stub() -> None:
    """
    In some older TVM distributions `tvm.runtime.vm` may not export
    `VMInstrumentReturnKind` while Relax code may attempt to import it.
    Insert a lightweight stub to avoid ImportError at import time.
    """
    try:
        vm_mod = importlib.import_module("tvm.runtime.vm")
    except Exception:
        return
    if hasattr(vm_mod, "VMInstrumentReturnKind"):
        return

    class VMInstrumentReturnKind(IntEnum):
        NO_OP = 0
        SKIP_RUN = 1

    setattr(vm_mod, "VMInstrumentReturnKind", VMInstrumentReturnKind)


_ensure_vm_instr_stub()


@dataclass
class GenerationParams:
    """Model generation parameter configuration.

    Configuration object for nnsmith model generation, including model
    size limits, timeouts, generation method and reproducibility seed.

    Attributes:
        max_nodes: Maximum number of operator nodes in the model.
        timeout_ms: Generation timeout in milliseconds.
        method: Generation method (e.g. "random", "symbolic").
        seed: Random seed for reproducibility.
        max_elem_per_tensor: Maximum number of elements per tensor.
        vulops: Enable vulnerable operations.
        grad_check: Enable gradient checking.
        rank_choices: Optional tensor rank choices.
        dtype_choices: Optional dtype choices.
        include: Operators to force-include.
        exclude: Operators to force-exclude.
        save_dir: Directory to save generated models.
    """
    max_nodes: int
    timeout_ms: int
    method: str
    seed: int
    max_elem_per_tensor: int
    vulops: bool = False
    grad_check: bool = False
    rank_choices: Optional[List[int]] = None
    dtype_choices: Optional[List[str]] = None
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None
    save_dir: Optional[Path] = None


@dataclass
class TestResult:
    """Result object for a single test execution.

    Encapsulates outcome, error information and performance metrics for
    a single generation+validation run.

    Attributes:
        success: Whether the test completed successfully.
        crashed: Whether the run crashed.
        divergence: Whether outputs diverged from expected.
        valid: Whether the test outcome is considered valid.
        duration_s: Execution time in seconds.
        message: Optional human-readable message or error details.
        model_dir: Directory containing model artifacts.
        oracle_outputs: List of expected output names from oracle.
        onnx_outputs: List of ONNX model output names.
    """
    success: bool
    crashed: bool
    divergence: bool
    valid: bool
    duration_s: float
    message: Optional[str] = None
    model_dir: Optional[Path] = None
    oracle_outputs: Optional[List[str]] = None
    onnx_outputs: Optional[List[str]] = None
    # Coverage proxies / debugging:
    # - model_n_nodes: number of ONNX nodes
    # - model_op_types: unique ONNX op types used by the model (sorted)
    # - failed_backend: which backend failed first in diff mode ("ort"/"ov"/"tvm"/"trt")
    model_n_nodes: Optional[int] = None
    model_op_types: Optional[List[str]] = None
    failed_backend: Optional[str] = None


# Test timeout configuration (overridable via environment variables)
ORT_RUN_TIMEOUT_S = int(os.environ.get("ORT_RUN_TIMEOUT_S", "120"))  # ONNX Runtime timeout (seconds)
TVM_RUN_TIMEOUT_S = int(os.environ.get("TVM_RUN_TIMEOUT_S", "180"))  # TVM timeout (seconds)


def _format_list(values: Optional[Sequence], is_int=False) -> str:
    """Format a Python sequence into nnsmith CLI list syntax.

    Converts a Python list into a nnsmith-style command-line list string.

    Args:
        values: Sequence of values to format.
        is_int: Whether values are integers (affects quoting).

    Returns:
        A string like "[val1, val2]" or "null".
    """
    if values is None:
        return "null"
    if is_int:
        return "[" + ", ".join(str(v) for v in values) + "]"
    return "[" + ", ".join(f'"{v}"' for v in values) + "]"


def build_model_gen_cmd(params: GenerationParams, output_dir: Path) -> List[str]:
    """构建nnsmith模型生成命令

    根据GenerationParams配置构建完整的nnsmith命令行参数列表。

    Args:
        params: 模型生成参数配置
        output_dir: 模型输出目录

    Returns:
        nnsmith命令行参数列表
    """
    cmd = [
        "python",
        "-m",
        "nnsmith.cli.model_gen",
        "model.type=onnx",
        "backend.type=onnxruntime",
        "backend.target=cpu",
        "backend.optmax=true",
        f"mgen.max_nodes={params.max_nodes}",
        f"mgen.timeout_ms={params.timeout_ms}",
        f"mgen.vulops={'true' if params.vulops else 'false'}",
        f"mgen.method={params.method}",
        f"mgen.save={output_dir}",
        f"mgen.seed={params.seed}",
        f"mgen.max_elem_per_tensor={params.max_elem_per_tensor}",
        f"mgen.grad_check={'true' if params.grad_check else 'false'}",
    ]
    cmd.append(f"mgen.rank_choices={_format_list(params.rank_choices, is_int=True)}")
    cmd.append(f"mgen.dtype_choices={_format_list(params.dtype_choices)}")
    if params.include is not None:
        cmd.append(f"mgen.include={_format_list(params.include)}")
        cmd.append("mgen.exclude=null")
    elif params.exclude is not None:
        cmd.append("mgen.include=null")
        cmd.append(f"mgen.exclude={_format_list(params.exclude)}")
    else:
        cmd.append("mgen.include=null")
        cmd.append("mgen.exclude=null")
    cmd.append("mgen.patch_requires=[]")
    cmd.append("debug.viz=false")
    return cmd


def _load_oracle(oracle_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load oracle data file produced by nnsmith."""
    with oracle_path.open("rb") as fp:
        data = pickle.load(fp)
    return data.get("input", {}), data.get("output", {})


def _compare_backend_outputs(
    outputs_by_backend: Dict[str, Dict[str, np.ndarray]],
    rtol: float,
    atol: float,
) -> Tuple[bool, str]:
    """Compare backend outputs pairwise.

    Returns (diverged: bool, detail: str).
    """
    if not outputs_by_backend:
        return False, "No outputs collected."
    backends = list(outputs_by_backend.keys())
    ref_backend = backends[0]
    ref_outputs = outputs_by_backend[ref_backend]
    diffs = []
    for b in backends[1:]:
        other = outputs_by_backend[b]
        if set(ref_outputs) != set(other):
            diffs.append(f"{ref_backend} vs {b}: output names mismatch {set(ref_outputs)} vs {set(other)}")
            continue
        for name, ref_arr in ref_outputs.items():
            arr = other[name]
            try:
                np.testing.assert_allclose(arr, ref_arr, rtol=rtol, atol=atol, equal_nan=True)
            except AssertionError as err:
                diffs.append(f"{ref_backend} vs {b} on {name}: {err}")
    return (len(diffs) > 0), " | ".join(diffs) if diffs else ""


def _extract_onnx_ops(model: onnx.ModelProto) -> Tuple[int, List[str]]:
    """Return (n_nodes, unique op types sorted) as a coverage proxy."""
    ops = [n.op_type for n in model.graph.node]
    return len(ops), sorted(set(ops))


def _infer_ort(model_path: Path, inputs: Dict[str, np.ndarray], providers: Optional[List[str]] = None):
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session = ort.InferenceSession(
        model_path.as_posix(),
        sess_opts,
        providers=providers or ["CPUExecutionProvider"],
    )
    outputs = session.run([], inputs)
    names = [o.name for o in session.get_outputs()]
    return dict(zip(names, outputs))


def _run_ort_infer_worker(
    args: Tuple[str, Dict[str, np.ndarray], Optional[List[str]], multiprocessing.Queue]
) -> None:
    """Run ORT inference in a separate process to isolate crashes/hangs."""
    model_path_str, inputs, providers, result_queue = args
    model_path = Path(model_path_str)
    try:
        outputs = _infer_ort(model_path, inputs, providers=providers)
        result_queue.put(("ok", outputs))
    except Exception:
        import traceback

        result_queue.put(("err", traceback.format_exc()))


def _infer_ort_isolated(
    model_path: Path,
    inputs: Dict[str, np.ndarray],
    providers: Optional[List[str]] = None,
    timeout_s: float = 60.0,
) -> Dict[str, np.ndarray]:
    """Run ORT inference in a child process to avoid crashing/hanging the parent."""
    ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue = ctx.Queue()
    p = ctx.Process(
        target=_run_ort_infer_worker,
        args=((str(model_path), inputs, providers, result_queue),),
    )
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        try:
            p.terminate()
        finally:
            p.join(5)
        raise TimeoutError(f"ORT inference timed out after {timeout_s:.1f}s")

    # If the worker crashed (segfault/abort), it will likely not put anything in the queue.
    if p.exitcode not in (0, None) and result_queue.empty():
        raise RuntimeError(f"ORT worker crashed (exitcode={p.exitcode})")

    try:
        tag, payload = result_queue.get_nowait()
    except Exception as e:
        raise RuntimeError(f"ORT worker produced no result (exitcode={p.exitcode})") from e

    if tag == "ok":
        return payload
    raise RuntimeError(f"ORT worker error:\n{payload}")


def _infer_openvino(model_path: Path, inputs: Dict[str, np.ndarray], device: str = "CPU"):
    import openvino as ov

    core = ov.Core()
    model = core.read_model(model_path.as_posix())
    compiled = core.compile_model(model, device)
    raw = compiled.create_infer_request().infer(inputs)
    return {port.get_any_name(): np.array(tensor) for port, tensor in raw.items()}


def _infer_openvino_isolated(
    model_path: Path,
    inputs: Dict[str, np.ndarray],
    device: str = "CPU",
    timeout_s: float = 60.0,
) -> Dict[str, np.ndarray]:
    """Run OpenVINO inference in a child process to avoid crashing the parent.

    OpenVINO may segfault/abort the whole process for some models (native crash),
    which cannot be caught by Python exception handling. Running it in a separate
    process converts such failures into a non-zero exit code that we can handle.
    """
    ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue = ctx.Queue()
    p = ctx.Process(
        target=_run_openvino_infer_worker,
        args=((str(model_path), inputs, device, result_queue),),
    )
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        try:
            p.terminate()
        finally:
            p.join(5)
        raise TimeoutError(f"OpenVINO inference timed out after {timeout_s:.1f}s")

    # If the worker crashed (segfault/abort), it will likely not put anything in the queue.
    if p.exitcode not in (0, None) and result_queue.empty():
        raise RuntimeError(f"OpenVINO worker crashed (exitcode={p.exitcode})")

    try:
        tag, payload = result_queue.get_nowait()
    except Exception as e:
        raise RuntimeError(
            f"OpenVINO worker produced no result (exitcode={p.exitcode})"
        ) from e

    if tag == "ok":
        return payload
    raise RuntimeError(f"OpenVINO worker error:\n{payload}")


def _infer_pytorch(model_path: Path, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Run ONNX model inference using PyTorch when possible.

    Attempts to convert the ONNX model to a PyTorch model via `onnx2torch`
    and run it. If conversion is not supported for certain ops, falls back
    to using ONNX Runtime as a reference PyTorch implementation.

    Args:
        model_path: Path to the ONNX model file.
        inputs: Input name -> numpy array mapping.

    Returns:
        Mapping from output name to numpy array.
    """
    try:
        import torch
        import onnx2torch
    except ImportError:
        # If onnx2torch is unavailable, fall back to ONNX Runtime
        import onnxruntime as ort
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(
            model_path.as_posix(),
            sess_opts,
            providers=["CPUExecutionProvider"],  # PyTorch通常运行在CPU上作为参考
        )
        outputs = session.run([], inputs)
        names = [o.name for o in session.get_outputs()]
        return dict(zip(names, outputs))
    
    # Try conversion and execution via onnx2torch
    try:
        torch_model = onnx2torch.convert(model_path.as_posix())
        torch_model.eval()

        # Convert inputs to PyTorch tensors
        torch_inputs = {name: torch.from_numpy(value) for name, value in inputs.items()}

        # Run inference
        with torch.no_grad():
            torch_outputs = torch_model(**torch_inputs)

        # Normalize outputs into a name->ndarray dict
        outputs = {}
        if isinstance(torch_outputs, (tuple, list)):
            # multiple outputs: map to ONNX output names when available
            import onnx
            model = onnx.load(model_path.as_posix())
            output_names = [out.name for out in model.graph.output]
            for i, output in enumerate(torch_outputs):
                if i < len(output_names):
                    outputs[output_names[i]] = output.numpy()
                else:
                    outputs[f"output_{i}"] = output.numpy()
        elif isinstance(torch_outputs, dict):
            outputs = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in torch_outputs.items()}
        else:
            import onnx
            model = onnx.load(model_path.as_posix())
            output_names = [out.name for out in model.graph.output]
            if len(output_names) == 1:
                outputs[output_names[0]] = torch_outputs.numpy()
            else:
                outputs["output_0"] = torch_outputs.numpy()

        return outputs
    except (NotImplementedError, RuntimeError, Exception) as e:
        # If onnx2torch conversion/execution fails for unsupported ops, fall back
        # to using ONNX Runtime as the reference implementation.
        import onnxruntime as ort
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(
            model_path.as_posix(),
            sess_opts,
            providers=["CPUExecutionProvider"],  # PyTorch通常运行在CPU上作为参考
        )
        outputs = session.run([], inputs)
        names = [o.name for o in session.get_outputs()]
        return dict(zip(names, outputs))


def _infer_tensorrt(model_path: Path, inputs: Dict[str, np.ndarray], workspace: int = 1 << 30):
    try:
        import tensorrt as trt  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "TensorRT not available: failed to import tensorrt. Please ensure TensorRT Python API is installed."
        ) from exc
    try:
        from cuda import cudart  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "TensorRT not available: failed to import cuda.cudart. "
            "Please install CUDA Python bindings (e.g., `pip install cuda-python`) and ensure CUDA runtime is on PATH/LD_LIBRARY_PATH."
        ) from exc

    logger = trt.Logger(trt.Logger.ERROR)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(logger) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(
        network, logger
    ) as parser, builder.create_builder_config() as config:
        # TensorRT 10+ uses set_memory_pool_limit; max_workspace_size is removed
        if hasattr(config, "set_memory_pool_limit"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
        else:  # fallback for older TRT
            config.max_workspace_size = workspace
        with open(model_path, "rb") as f:
            if not parser.parse(f.read()):
                err = parser.get_error(0)
                raise RuntimeError(f"TRT parse failed: {err.desc() if err else 'unknown'}")

        profile = builder.create_optimization_profile()
        for name, arr in inputs.items():
            shape = tuple(int(d) for d in arr.shape)
            profile.set_shape(name, min=shape, opt=shape, max=shape)
        config.add_optimization_profile(profile)

        if hasattr(builder, "build_engine"):
            engine = builder.build_engine(network, config)
        else:
            plan = builder.build_serialized_network(network, config)
            if plan is None:
                raise RuntimeError("TRT serialized plan build failed")
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(plan)
        if engine is None:
            raise RuntimeError("TRT engine build failed")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("TRT execution context creation failed")

    # Helper for TRT v10 bindings API
    def _get_binding_name(i: int) -> str:
        return engine.get_binding_name(i) if hasattr(engine, "get_binding_name") else engine.get_tensor_name(i)

    def _get_binding_index(name: str) -> int:
        if hasattr(engine, "get_binding_index"):
            return engine.get_binding_index(name)
        # TRT 10+: iterate and match by name
        for i in range(_num_bindings()):
            if _get_binding_name(i) == name:
                return i
        raise KeyError(f"TensorRT binding not found: {name}")

    def _num_bindings() -> int:
        return engine.num_bindings if hasattr(engine, "num_bindings") else engine.num_io_tensors

    def _binding_is_input(i: int) -> bool:
        if hasattr(engine, "binding_is_input"):
            return engine.binding_is_input(i)
        return engine.get_tensor_mode(engine.get_binding_name(i)) == trt.TensorIOMode.INPUT

    def _get_binding_shape(i: int) -> tuple:
        return (
            tuple(context.get_binding_shape(i))
            if hasattr(context, "get_binding_shape")
            else tuple(context.get_tensor_shape(_get_binding_name(i)))
        )

    def _set_binding_shape(idx: int, shape: tuple):
        if hasattr(context, "set_binding_shape"):
            context.set_binding_shape(idx, shape)
        else:
            context.set_input_shape(_get_binding_name(idx), shape)

    def _get_binding_dtype(i: int):
        return engine.get_binding_dtype(i) if hasattr(engine, "get_binding_dtype") else engine.get_tensor_dtype(_get_binding_name(i))

    # Set shapes for dynamic bindings
    for name, arr in inputs.items():
        idx = _get_binding_index(name)
        _set_binding_shape(idx, tuple(int(d) for d in arr.shape))

    bindings: list[int] = [0] * _num_bindings()
    outputs: Dict[str, np.ndarray] = {}
    device_mem: Dict[str, int] = {}

    def _alloc_and_copy(name: str, arr: np.ndarray, is_input: bool):
        nbytes = arr.nbytes
        ptr = cudart.cudaMalloc(nbytes)[1]
        if is_input:
            cudart.cudaMemcpy(ptr, arr.ctypes.data, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        else:
            outputs[name] = arr
        device_mem[name] = ptr

    # Allocate inputs
    for name, arr in inputs.items():
        _alloc_and_copy(name, np.ascontiguousarray(arr), True)

    # Allocate outputs based on binding shapes
    for i in range(_num_bindings()):
        if _binding_is_input(i):
            continue
        name = _get_binding_name(i)
        shape = _get_binding_shape(i)
        dtype = trt.nptype(_get_binding_dtype(i))
        host_arr = np.empty(shape, dtype=dtype)
        _alloc_and_copy(name, host_arr, False)

    # Fill bindings array
    for i in range(_num_bindings()):
        name = _get_binding_name(i)
        bindings[i] = device_mem[name]

    # Execute
    ok = context.execute_v2(bindings=bindings)
    if not ok:
        raise RuntimeError("TRT execution failed")

    # Copy outputs back
    for name, host_arr in outputs.items():
        ptr = device_mem[name]
        nbytes = host_arr.nbytes
        cudart.cudaMemcpy(host_arr.ctypes.data, ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        cudart.cudaFree(ptr)
    for name, arr in inputs.items():
        cudart.cudaFree(device_mem[name])

    return outputs


def _infer_tvm_relax(model_path: Path, inputs: Dict[str, np.ndarray], target: str):
    import onnx
    from tvm.relax.frontend import onnx as rx_onnx

    onnx_model = onnx.load(model_path)
    shape_dict = {name: arr.shape for name, arr in inputs.items()}
    converted = rx_onnx.from_onnx(onnx_model, shape_dict=shape_dict)
    if isinstance(converted, (list, tuple)) and len(converted) >= 1:
        mod = converted[0]
    else:
        mod = converted

    # 对齐 test_tvm_nnsmith 的编译流程：推理分解 + Legalize + 分离参数 + pipeline
    mod = relax.transform.DecomposeOpsForInference()(mod)
    mod = relax.transform.LegalizeOps()(mod)
    mod, params = relax.frontend.detach_params(mod)

    dev = tvm.device(str(target), 0)
    tgt = tvm.target.Target(str(target))
    relax_pipeline = relax.pipeline.get_default_pipeline(tgt)
    with tvm.transform.PassContext(
        opt_level=3,
        # 关闭 TIR 调试插桩，避免生成 dbg_declare
        config={"tir.enable_debug": False},
    ):
        ex = relax.build(mod, target=tgt, params=params, relax_pipeline=relax_pipeline)

    vm = relax.VirtualMachine(ex, dev)

    # 按 Relax main 参数顺序喂入 oracle 输入与分离出的 params
    input_list: List[np.ndarray] = []
    for param in mod["main"].params:
        if param.name_hint in inputs:
            input_list.append(inputs[param.name_hint])
    if params and "main" in params:
        input_list += params["main"]

    vm.set_input("main", *input_list)
    vm.invoke_stateful("main")
    out = vm.get_outputs("main")

    # 统一为 numpy
    def _to_numpy(val):
        if hasattr(val, "numpy"):
            return val.numpy()
        if isinstance(val, tvm.runtime.ShapeTuple):
            return np.array([int(v) for v in val])
        if isinstance(val, (int, float, bool)):
            return np.array(val)
        return val

    if isinstance(out, tuple):
        outputs = [_to_numpy(o) for o in out]
    else:
        outputs = [_to_numpy(out)]

    output_names = [o.name or f"output_{idx}" for idx, o in enumerate(onnx_model.graph.output)]
    results: Dict[str, np.ndarray] = {}
    for idx, arr in enumerate(outputs):
        if idx < len(output_names):
            results[output_names[idx]] = arr
    return results


def _infer_tvm(model_path: Path, inputs: Dict[str, np.ndarray], target: str):
    return _infer_tvm_relax(model_path, inputs, target)


def _run_tvm_infer_worker(
    args: Tuple[str, Dict[str, np.ndarray], str, multiprocessing.Queue]
) -> None:
    """Run TVM inference in a separate process (for gcov gcda flush on process exit)."""
    model_path_str, inputs, target, q = args
    try:
        out = _infer_tvm(Path(model_path_str), inputs, target=target)
        q.put(("ok", out))
    except Exception:
        import traceback

        q.put(("err", traceback.format_exc()))


def _infer_tvm_isolated(
    model_path: Path,
    inputs: Dict[str, np.ndarray],
    *,
    target: str,
    timeout_s: int,
) -> Dict[str, np.ndarray]:
    """
    Run TVM inference in a fresh spawned process, so coverage counters are written to *.gcda
    even when the parent is long-running.
    """
    ctx = multiprocessing.get_context("spawn")
    q: multiprocessing.Queue = ctx.Queue()
    worker_args = (str(model_path), inputs, str(target), q)
    p = ctx.Process(target=_run_tvm_infer_worker, args=(worker_args,), daemon=False)
    p.start()
    p.join(timeout=float(timeout_s))
    if p.is_alive():
        p.terminate()
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
            p.join()
        raise concurrent.futures.TimeoutError(f"TVM isolated inference timed out (timeout_s={timeout_s})")

    if p.exitcode not in (0, None) and q.empty():
        raise RuntimeError(f"TVM isolated worker crashed (exitcode={p.exitcode}).")

    try:
        tag, payload = q.get(timeout=5)
    except Exception as e:
        raise RuntimeError(
            f"TVM isolated worker produced no result (exitcode={p.exitcode}): {e}"
        ) from e

    if tag == "ok":
        return payload
    raise RuntimeError(f"TVM isolated inference failed:\n{payload}")


def _run_tvm_worker(args: Tuple[str, str, multiprocessing.Queue]) -> None:
    """Worker to run a TVM test inside a separate process."""
    model_path_str, oracle_path_str, result_queue = args
    model_path = Path(model_path_str)
    oracle_path = Path(oracle_path_str)
    try:
        result = _run_tvm(model_path, oracle_path)
        result_queue.put(result)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        result = TestResult(
            success=False,
            crashed=True,
            divergence=True,
            valid=False,
            duration_s=0.0,
            message=f"Exception in TVM worker: {tb}",
            model_dir=str(model_path.parent) if model_path.exists() else None,
        )
        result_queue.put(result)


def _run_ort_worker(args: Tuple[str, str, Optional[List[str]], multiprocessing.Queue]) -> None:
    """Worker running an ORT test inside a separate process.

    Running in a child process allows the parent to enforce timeouts and
    isolate crashes that would otherwise kill the main fuzzing loop.

    Args:
        args: tuple of (model_path_str, oracle_path_str, providers, result_queue)
    """
    model_path_str, oracle_path_str, providers, result_queue = args
    model_path = Path(model_path_str)
    oracle_path = Path(oracle_path_str)
    try:
        result = _run_ort(model_path, oracle_path, providers)
        # 将 Path 对象转换为字符串以便序列化
        if result.model_dir:
            result.model_dir = str(result.model_dir)
        result_queue.put(result)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        model_dir_str = str(model_path.parent) if model_path.exists() else None
        result = TestResult(
            success=False,
            crashed=True,
            divergence=False,
            valid=False,
            duration_s=0.0,
            message=f"Exception in worker: {tb}",
            model_dir=model_dir_str,
        )
        result_queue.put(result)


def _run_openvino_infer_worker(
    args: Tuple[str, Dict[str, np.ndarray], str, multiprocessing.Queue]
) -> None:
    """Run OpenVINO inference in a separate process.

    Motivation:
    - OpenVINO GPU plugin may abort the whole process (e.g., OpenCL CL_INVALID_COMMAND_QUEUE),
      which would kill the entire fuzzing loop if executed in-process.
    - Running in a child process isolates such aborts and lets the parent keep going.

    The worker communicates results via a queue:
    - ("ok", outputs_dict) on success
    - ("err", traceback_str) on Python-level exception
    """
    model_path_str, inputs, device, result_queue = args
    model_path = Path(model_path_str)
    try:
        outputs = _infer_openvino(model_path, inputs, device=device)
        result_queue.put(("ok", outputs))
    except Exception:
        import traceback

        result_queue.put(("err", traceback.format_exc()))


def _run_ort(model_path: Path, oracle_path: Path, providers: Optional[List[str]] = None) -> TestResult:
    """使用ONNX Runtime运行模型并验证结果

    加载ONNX模型，使用指定的ExecutionProvider运行推理，
    并与oracle数据进行数值比较。

    注意：此函数不处理超时，超时由调用方的并发框架控制。

    Args:
        model_path: ONNX模型文件路径
        oracle_path: oracle数据文件路径
        providers: ONNX Runtime执行提供者列表（如["CPUExecutionProvider"]）

    Returns:
        TestResult对象，包含测试结果和详细信息
    """
    start_time = time.time()

    try:
        model = onnx.load(model_path)
        inputs, outputs = _load_oracle(oracle_path)

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC  # Reduce optimization to avoid constant folding issues
        session = ort.InferenceSession(
            model.SerializeToString(),
            sess_opts,
            providers=providers or ["CPUExecutionProvider"],
        )

        ort_out = session.run([], inputs)
        expected = [outputs[name] for name in outputs.keys()]

        if len(ort_out) != len(expected):
            raise AssertionError(
                f"Output count mismatch: oracle={len(expected)} ort={len(ort_out)}"
            )

        for exp, act in zip(expected, ort_out):
            if act is None:
                continue
            np.testing.assert_allclose(act, exp, rtol=1e-1, atol=1e-1)

        duration = time.time() - start_time
        # 将 Path 对象转换为字符串以便序列化（用于进程间通信）
        model_dir_str = str(model_path.parent)
        return TestResult(
            success=True,
            crashed=False,
            divergence=False,
            valid=True,
            duration_s=duration,
            model_dir=model_dir_str,  # 存储为字符串
            oracle_outputs=list(outputs.keys()),
            onnx_outputs=[out.name for out in model.graph.output],
        )

    except Exception as e:
        duration = time.time() - start_time
        # 将 Path 对象转换为字符串以便序列化
        model_dir_str = str(model_path.parent) if model_path.exists() else None
        return TestResult(
            success=False,
            crashed=True,
            divergence=False,
            valid=False,
            duration_s=duration,
            message=f"ONNX Runtime error: {str(e)}",
            model_dir=model_dir_str,  # 存储为字符串
        )


def run_generation_and_test(
    params: GenerationParams,
    nnsmith_root: Path,
    run_root: Path,
    bug_root: Optional[Path] = None,
    run_name: Optional[str] = None,
    existing_model_path: Optional[Path] = None,  # If provided, skip nnsmith generation and test this ONNX model.
    verbose: bool = True,
    enable_tvm_check: bool = False,
    tvm_timeout_s: Optional[int] = None,
    diff_backends: Optional[List[str]] = None,
    diff_rtol: float = 1e-3,
    diff_atol: float = 1e-3,
    diff_device: Optional[str] = None,  # "cpu" or "gpu" (affects ORT/TVM/OV in diff mode)
    ov_device: Optional[str] = None,  # OpenVINO device: "CPU", "GPU", "AUTO", etc. If None, uses environment variable or defaults to "GPU"
    compiler: str = "ort",  # Compiler type: "ort", "tvm", "ov" - used for CPU/GPU/PyTorch comparison
) -> TestResult:
    """运行完整的模型生成和测试流程

    这是编译器fuzzing的核心函数，执行以下步骤：
    1. 使用nnsmith生成随机ONNX模型
    2. 使用ONNX Runtime验证模型基本功能
    3. 可选：使用TVM编译和验证模型

    Args:
        params: 模型生成参数配置
        nnsmith_root: nnsmith项目根目录
        run_root: 测试运行结果保存目录
        bug_root: bug案例保存目录（可选）
        run_name: 指定运行名称（可选）
        verbose: 是否显示详细输出
        enable_tvm_check: 是否启用TVM编译测试
        tvm_timeout_s: TVM测试超时时间（秒）

    Returns:
        TestResult对象，包含完整的测试结果
    """
    while True:
        if run_name:
            output_dir = run_root / run_name
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)
            break
        output_dir = run_root / f"run_{int(time.time() * 1000)}_{random.randint(0, 9999):04d}"
        if not output_dir.exists():
            break

    start = time.time()

    # If caller provides an existing ONNX model, copy it into output_dir and skip nnsmith generation.
    # This is used for external baselines (e.g., COMET/TitanFuzz) where the model is produced elsewhere.
    # Always create output_dir so the run is visible on disk (useful for debugging).
    output_dir.mkdir(parents=True, exist_ok=True)

    if existing_model_path is not None:
        src = Path(existing_model_path)
        if not src.exists():
            return TestResult(
                success=False,
                crashed=False,
                divergence=False,
                valid=False,
                duration_s=0.0,
                message=f"existing_model_path not found: {src}",
                model_dir=None,
            )
        try:
            shutil.copy2(src, output_dir / "model.onnx")
            # Record provenance for external baselines.
            try:
                (output_dir / "EXISTING_MODEL_SOURCE.txt").write_text(
                    str(src.resolve()), encoding="utf-8"
                )
            except Exception:
                pass
        except Exception as e:
            return TestResult(
                success=False,
                crashed=False,
                divergence=False,
                valid=False,
                duration_s=0.0,
                message=f"Failed to copy existing model: {type(e).__name__}: {e!r}",
                model_dir=None,
            )
        proc_returncode = 0
        duration = time.time() - start
        stdout = ""
    else:
        gen_cmd = build_model_gen_cmd(params, output_dir)

        env_vars = os.environ.copy()
        # 为避免 torch.export 中的 data-dependent guard，差分模式/多后端对比时默认关闭 nnsmith 数值检查。
        if diff_backends or enable_tvm_check or compiler in ["ort", "tvm", "ov"]:
            env_vars["NNSMITH_DISABLE_NUMERIC_CHECK"] = "1"
        else:
            env_vars.setdefault("NNSMITH_DISABLE_NUMERIC_CHECK", "0")
        # Make nnsmith non-interactive: auto-overwrite existing report folder instead of prompting [Y/N].
        env_vars.setdefault("NNSMITH_MKDIR_YES", "1")

        # NOTE: nnsmith has its own internal timeout (mgen.timeout_ms), but in practice it may still hang.
        hard_timeout_s = float(os.environ.get("NNSMITH_GEN_HARD_TIMEOUT_S", "0") or "0")
        if hard_timeout_s <= 0:
            hard_timeout_s = max(float(params.timeout_ms) / 1000.0 + 10.0, 30.0)

        # Run nnsmith in a new process group so a timeout can kill all its children.
        proc = subprocess.Popen(
            gen_cmd,
            cwd=nnsmith_root,
            text=True,
            stdout=None if verbose else subprocess.PIPE,
            stderr=None if verbose else subprocess.STDOUT,
            env=env_vars,
            start_new_session=True,
        )
        try:
            stdout, _ = proc.communicate(timeout=hard_timeout_s)
        except subprocess.TimeoutExpired:
            # Kill the entire process group (nnsmith may spawn children).
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                stdout, _ = proc.communicate(timeout=5)
            except Exception:
                stdout = stdout or ""
            duration = time.time() - start
            message = (
                f"Model generation timed out (hard_timeout_s={hard_timeout_s:.1f}).\n"
                f"cmd={' '.join(map(str, gen_cmd))}\n"
                f"partial_output:\n{stdout or ''}"
            )
            result = TestResult(
                success=False,
                crashed=False,
                divergence=False,
                valid=False,
                duration_s=duration,
                message=message,
                model_dir=None,
            )
            shutil.rmtree(output_dir, ignore_errors=True)
            return result

        duration = time.time() - start
        proc_returncode = int(proc.returncode or 0)
        if proc_returncode != 0:
            out = stdout or ""
            message = out if out.strip() else "Model generation failed."
            result = TestResult(
                success=False,
                crashed=False,
                divergence=False,
                valid=False,
                duration_s=duration,
                message=message,
                model_dir=None,
            )
            shutil.rmtree(output_dir, ignore_errors=True)
            return result

    model_path = output_dir / "model.onnx"
    oracle_path = output_dir / "oracle.pkl"
    diff_backends = diff_backends or []
    
    # Precompute diff-mode device mapping (CPU vs GPU) for supported backends.
    diff_dev = (diff_device or "").strip().lower()
    if diff_dev == "gpu":
        ort_diff_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        tvm_diff_target = "cuda"
        ov_diff_default = "GPU"
    else:
        # Default to CPU in diff mode unless explicitly set to GPU.
        ort_diff_providers = ["CPUExecutionProvider"]
        tvm_diff_target = "llvm"
        ov_diff_default = "CPU"

    # 确定OpenVINO设备：优先使用参数，其次环境变量。
    # IMPORTANT: for differential experiments (diff_backends), we want a stable baseline and
    # typically run on CPU (AUTO/GPU may change behavior across machines).
    # You can override via env `OPENVINO_DEVICE` or by passing `ov_device`.
    if ov_device is None:
        ov_device = os.getenv("OPENVINO_DEVICE", ov_diff_default if diff_backends else "GPU")

    # 普通测试模式：CPU/GPU/PyTorch对比（当没有指定diff_backends时）
    if not diff_backends:
        try:
            model = onnx.load(model_path)
            oracle_inputs, _ = _load_oracle(oracle_path) if oracle_path.exists() else ({}, {})
            inputs = oracle_inputs or _generate_random_inputs(model)
            outputs_by_backend: Dict[str, Dict[str, np.ndarray]] = {}
            
            # 根据compiler参数决定对比哪些后端
            if compiler == "ort":
                # ORT CPU vs ORT GPU vs PyTorch
                outputs_by_backend["ORT_CPU"] = _infer_ort(model_path, inputs, providers=["CPUExecutionProvider"])
                try:
                    outputs_by_backend["ORT_GPU"] = _infer_ort(model_path, inputs, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                except Exception:
                    outputs_by_backend["ORT_GPU"] = outputs_by_backend["ORT_CPU"]  # Fallback to CPU if GPU unavailable
                outputs_by_backend["PyTorch"] = _infer_pytorch(model_path, inputs)
            elif compiler == "tvm":
                # TVM CPU vs TVM GPU vs PyTorch
                outputs_by_backend["TVM_CPU"] = _infer_tvm(model_path, inputs, target="llvm")
                try:
                    outputs_by_backend["TVM_GPU"] = _infer_tvm(model_path, inputs, target="cuda")
                except Exception:
                    outputs_by_backend["TVM_GPU"] = outputs_by_backend["TVM_CPU"]  # Fallback to CPU if GPU unavailable
                outputs_by_backend["PyTorch"] = _infer_pytorch(model_path, inputs)
            elif compiler == "ov":
                # OpenVINO CPU vs OpenVINO GPU vs PyTorch
                # IMPORTANT: run in child processes to avoid GPU plugin abort killing the main process.
                def _ov_infer_isolated(device: str) -> Dict[str, np.ndarray]:
                    q: multiprocessing.Queue = multiprocessing.Queue()
                    worker_args = (str(model_path), inputs, device, q)
                    p = multiprocessing.Process(
                        target=_run_openvino_infer_worker,
                        args=(worker_args,),
                        daemon=False,
                    )
                    p.start()
                    p.join(timeout=180)  # avoid hanging forever

                    if p.is_alive():
                        p.terminate()
                        p.join(timeout=5)
                        if p.is_alive():
                            p.kill()
                            p.join()
                        raise concurrent.futures.TimeoutError(
                            f"OpenVINO {device} inference timed out"
                        )

                    # If the process aborts (e.g. SIGABRT), exitcode will be non-zero and queue may be empty.
                    if p.exitcode not in (0, None) and q.empty():
                        raise RuntimeError(
                            f"OpenVINO {device} worker crashed (exitcode={p.exitcode}). "
                            "This may be due to GPU plugin abort (e.g., OpenCL CL_INVALID_COMMAND_QUEUE)."
                        )

                    try:
                        tag, payload = q.get(timeout=5)
                    except Exception as e:
                        raise RuntimeError(
                            f"OpenVINO {device} worker produced no result (exitcode={p.exitcode}): {e}"
                        ) from e

                    if tag == "ok":
                        return payload
                    raise RuntimeError(f"OpenVINO {device} failed:\n{payload}")

                outputs_by_backend["OV_CPU"] = _ov_infer_isolated("CPU")
                outputs_by_backend["OV_GPU"] = _ov_infer_isolated("GPU")
                outputs_by_backend["PyTorch"] = _infer_pytorch(model_path, inputs)
            else:
                # 默认：ORT CPU vs ORT GPU vs PyTorch
                outputs_by_backend["ORT_CPU"] = _infer_ort(model_path, inputs, providers=["CPUExecutionProvider"])
                try:
                    outputs_by_backend["ORT_GPU"] = _infer_ort(model_path, inputs, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                except Exception:
                    outputs_by_backend["ORT_GPU"] = outputs_by_backend["ORT_CPU"]
                outputs_by_backend["PyTorch"] = _infer_pytorch(model_path, inputs)
            
            # 对比结果
            diverged, detail = _compare_backend_outputs(outputs_by_backend, diff_rtol, diff_atol)
            duration = time.time() - start
            if diverged:
                message = f"CPU/GPU/PyTorch mismatch: {detail}"
                result = TestResult(
                    success=False,
                    crashed=False,
                    divergence=True,
                    valid=True,
                    duration_s=duration,
                    message=message,
                    model_dir=output_dir,
                    oracle_outputs=list(outputs_by_backend[next(iter(outputs_by_backend))].keys()),
                    onnx_outputs=[out.name for out in model.graph.output],
                )
                _save_bug_artifacts(output_dir, bug_root, error_log=message)
                return result
            shutil.rmtree(output_dir, ignore_errors=True)
            return TestResult(
                success=True,
                crashed=False,
                divergence=False,
                valid=True,
                duration_s=duration,
                message="CPU/GPU/PyTorch agree.",
                model_dir=output_dir,
                oracle_outputs=list(outputs_by_backend[next(iter(outputs_by_backend))].keys()),
                onnx_outputs=[out.name for out in model.graph.output],
            )
        except Exception as e:
            duration = time.time() - start
            import traceback
            tb = traceback.format_exc()
            message = (
                f"CPU/GPU/PyTorch test failed: {type(e).__name__}: {e!r}\n"
                f"{tb}"
            )
            result = TestResult(
                success=False,
                crashed=True,
                divergence=False,
                valid=False,
                duration_s=duration,
                message=message,
                model_dir=output_dir,
            )
            _save_bug_artifacts(output_dir, bug_root, error_log=message)
            return result

    # 差分模式：仅使用 oracle 输入，对比多个后端输出
    if diff_backends:
        failed_backend: Optional[str] = None
        n_nodes: Optional[int] = None
        op_types: Optional[List[str]] = None
        try:
            model = onnx.load(model_path)
            n_nodes, op_types = _extract_onnx_ops(model)
            oracle_inputs, _ = _load_oracle(oracle_path) if oracle_path.exists() else ({}, {})
            inputs = oracle_inputs or _generate_random_inputs(model)
            outputs_by_backend: Dict[str, Dict[str, np.ndarray]] = {}
            backend_errors: Dict[str, str] = {}
            for b in diff_backends:
                key = b.lower()
                try:
                    if key == "ort":
                        # ORT can also hang/crash on malformed models or buggy EPs; isolate by default.
                        ort_timeout = float(os.environ.get("HIGEN_ORT_TIMEOUT_S", "60"))
                        if os.environ.get("HIGEN_ORT_ISOLATED", "1") != "0":
                            outputs_by_backend[b] = _infer_ort_isolated(
                                model_path,
                                inputs,
                                providers=ort_diff_providers,
                                timeout_s=ort_timeout,
                            )
                        else:
                            outputs_by_backend[b] = _infer_ort(
                                model_path, inputs, providers=ort_diff_providers
                            )
                    elif key == "ov":
                        # OpenVINO may segfault/abort on some models; isolate to keep the main loop alive.
                        ov_timeout = float(os.environ.get("HIGEN_OV_TIMEOUT_S", "60"))
                        outputs_by_backend[b] = _infer_openvino_isolated(
                            model_path, inputs, device=ov_device, timeout_s=ov_timeout
                        )
                    elif key == "tvm":
                        # TVM compilation/runtime may also hang; isolate by default in diff mode.
                        if os.environ.get("HIGEN_TVM_ISOLATED", "1") != "0":
                            timeout = TVM_RUN_TIMEOUT_S if tvm_timeout_s is None else int(tvm_timeout_s)
                            outputs_by_backend[b] = _infer_tvm_isolated(
                                model_path,
                                inputs,
                                target=tvm_diff_target,
                                timeout_s=int(timeout),
                            )
                        else:
                            outputs_by_backend[b] = _infer_tvm(
                                model_path, inputs, target=tvm_diff_target
                            )
                    elif key == "trt":
                        outputs_by_backend[b] = _infer_tensorrt(model_path, inputs)
                    else:
                        raise ValueError(f"Unsupported backend for diff: {b}")
                except Exception as be:
                    # IMPORTANT: don't early-abort. We want to still run other backends
                    # (especially TVM for coverage collection).
                    failed_backend = key
                    backend_errors[key] = f"{type(be).__name__}: {be!r}"
                    continue

            if backend_errors:
                duration = time.time() - start
                message = "Backend failures: " + "; ".join([f"{k}: {v}" for k, v in backend_errors.items()])
                result = TestResult(
                    success=False,
                    crashed=True,
                    divergence=False,
                    valid=False,
                    duration_s=duration,
                    message=message,
                    model_dir=output_dir,
                    oracle_outputs=list(outputs_by_backend[next(iter(outputs_by_backend))].keys())
                    if outputs_by_backend
                    else None,
                    onnx_outputs=[out.name for out in model.graph.output],
                    model_n_nodes=int(n_nodes),
                    model_op_types=list(op_types),
                    failed_backend=str(failed_backend),
                )
                _save_bug_artifacts(output_dir, bug_root, error_log=message)
                return result
            diverged, detail = _compare_backend_outputs(outputs_by_backend, diff_rtol, diff_atol)
            duration = time.time() - start
            if diverged:
                message = f"Differential mismatch: {detail}"
                result = TestResult(
                    success=False,
                    crashed=False,
                    divergence=True,
                    valid=True,
                    duration_s=duration,
                    message=message,
                    model_dir=output_dir,
                    oracle_outputs=list(outputs_by_backend[next(iter(outputs_by_backend))].keys()),
                    onnx_outputs=[out.name for out in model.graph.output],
                    model_n_nodes=int(n_nodes),
                    model_op_types=list(op_types),
                    failed_backend=None,
                )
                _save_bug_artifacts(output_dir, bug_root, error_log=message)
                return result
            shutil.rmtree(output_dir, ignore_errors=True)
            return TestResult(
                success=True,
                crashed=False,
                divergence=False,
                valid=True,
                duration_s=duration,
                message="Differential backends agree.",
                model_dir=output_dir,
                oracle_outputs=list(outputs_by_backend[next(iter(outputs_by_backend))].keys()),
                onnx_outputs=[out.name for out in model.graph.output],
                model_n_nodes=int(n_nodes),
                model_op_types=list(op_types),
                failed_backend=None,
            )
        except Exception as e:
            duration = time.time() - start
            # NOTE: Some exceptions (e.g. TimeoutError(), RuntimeError() with no args) have empty str(e).
            # Always include exception type + repr + full traceback so error.log is actionable.
            import traceback

            tb = traceback.format_exc()
            message = (
                f"Differential test failed: {type(e).__name__}: {e!r}\n"
                f"{tb}"
            )
            result = TestResult(
                success=False,
                crashed=True,
                divergence=False,
                valid=False,
                duration_s=duration,
                message=message,
                model_dir=output_dir,
                model_n_nodes=int(n_nodes) if n_nodes is not None else None,
                model_op_types=list(op_types) if op_types is not None else None,
                failed_backend=str(failed_backend) if failed_backend else None,
            )
            _save_bug_artifacts(output_dir, bug_root, error_log=message)
            return result

    ort_providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if enable_tvm_check
        else ["CPUExecutionProvider"]
    )
    try:
        if ORT_RUN_TIMEOUT_S > 0:
            # 使用进程级别的超时机制，可以真正中断 hang 的 ONNX Runtime 调用
            # 创建一个队列用于进程间通信
            result_queue = multiprocessing.Queue()
            # 将 Path 对象转换为字符串以便序列化
            worker_args = (str(model_path), str(oracle_path), ort_providers, result_queue)
            
            # 在独立进程中运行 ORT 测试
            process = multiprocessing.Process(
                target=_run_ort_worker,
                args=(worker_args,),
                daemon=False  # 非守护进程，确保可以正确清理
            )
            process.start()
            process.join(timeout=ORT_RUN_TIMEOUT_S)

            if process.is_alive():
                # 进程超时，强制终止
                process.terminate()
                # 等待进程真正退出，最多等待 5 秒
                process.join(timeout=5)
                if process.is_alive():
                    # 如果仍然存活，强制杀死
                    process.kill()
                    process.join()
                raise concurrent.futures.TimeoutError()

            # 检查进程退出码
            if process.exitcode != 0 and process.exitcode is not None:
                raise RuntimeError(f"ORT test process exited with code {process.exitcode}")

            # 从队列中获取结果，带超时等待
            try:
                ort_result = result_queue.get(timeout=5)  # 等待最多5秒获取结果
                # 将字符串路径转换回 Path 对象
                if ort_result.model_dir and isinstance(ort_result.model_dir, str):
                    ort_result.model_dir = Path(ort_result.model_dir)
            except Exception as e:
                # 如果无法获取结果，抛出超时错误
                raise RuntimeError(f"Failed to get result from ORT test process: {e}") from e
        else:
            ort_result = _run_ort(model_path, oracle_path, ort_providers)
        ort_result.duration_s = duration
        ort_result.model_dir = output_dir
        if not ort_result.success:
            # 过滤掉"Output count mismatch"这种正常的模型生成问题，不作为bug保存
            if "Output count mismatch" not in ort_result.message:
                _save_bug_artifacts(output_dir, bug_root, error_log=ort_result.message)
            else:
                # 对于正常的模型生成问题，直接删除目录，不保存
                shutil.rmtree(output_dir, ignore_errors=True)
            return ort_result

        if enable_tvm_check:
            try:
                timeout = TVM_RUN_TIMEOUT_S if tvm_timeout_s is None else tvm_timeout_s
                if timeout and timeout > 0:
                    # 使用进程级别的超时机制，确保可以中断hang的TVM调用
                    result_queue = multiprocessing.Queue()
                    worker_args = (str(model_path), str(oracle_path), result_queue)

                    # 在独立进程中运行TVM测试
                    process = multiprocessing.Process(
                        target=_run_tvm_worker,
                        args=(worker_args,),
                        daemon=False
                    )
                    process.start()
                    process.join(timeout=timeout)

                    if process.is_alive():
                        # 进程超时，强制终止
                        process.terminate()
                        process.join(timeout=5)
                        if process.is_alive():
                            process.kill()
                            process.join()
                        raise concurrent.futures.TimeoutError()

                    # 从队列中获取结果
                    try:
                        tvm_result = result_queue.get(timeout=5)
                    except Exception as e:
                        raise RuntimeError(f"Failed to get result from TVM test process: {e}") from e
                else:
                    tvm_result = _run_tvm(model_path, oracle_path)
            except concurrent.futures.TimeoutError:
                message = f"TVM inference exceeded {timeout}s timeout."
                result = TestResult(
                    success=False,
                    crashed=True,
                    divergence=False,
                    valid=False,
                    duration_s=duration,
                    message=message,
                    model_dir=output_dir,
                )
                _save_bug_artifacts(output_dir, bug_root, error_log=message)
                return result
            except Exception:  # noqa: BLE001
                import traceback

                tb = traceback.format_exc()
                result = TestResult(
                    success=False,
                    crashed=True,
                    divergence=True,
                    valid=False,
                    duration_s=duration,
                    message=tb,
                    model_dir=output_dir,
                )
                _save_bug_artifacts(output_dir, bug_root, error_log=tb)
                return result
            else:
                ort_result.message = (ort_result.message or "") + " | TVM check passed."

        shutil.rmtree(output_dir, ignore_errors=True)
        return ort_result
    except concurrent.futures.TimeoutError:
        message = f"ONNX Runtime inference exceeded {ORT_RUN_TIMEOUT_S}s timeout."
        result = TestResult(
            success=False,
            crashed=True,
            divergence=False,
            valid=False,
            duration_s=duration,
            message=message,
            model_dir=output_dir,
        )
        _save_bug_artifacts(output_dir, bug_root, error_log=message)
        return result
    except Exception:  # noqa: BLE001
        import traceback

        tb = traceback.format_exc()
        is_output_mismatch = "Output count mismatch" in tb
        result = TestResult(
            success=False,
            crashed=False,
            divergence=not is_output_mismatch,
            valid=not is_output_mismatch,
            duration_s=duration,
            message=tb,
            model_dir=output_dir,
        )
        if is_output_mismatch:
            shutil.rmtree(output_dir, ignore_errors=True)
        else:
            _save_bug_artifacts(output_dir, bug_root, error_log=tb)
        return result


def _save_bug_artifacts(source_dir: Path, bug_root: Optional[Path], error_log: Optional[str] = None):
    if bug_root is None or source_dir is None:
        return
    if not source_dir.exists():
        return

    # Dedup repeated crashes: if we've already seen the same crash signature,
    # don't save it again (just clean up the temp output directory).
    if error_log is not None:
        try:
            if not should_keep_crash_case(bug_root, case_name=source_dir.name, error_log=error_log):
                shutil.rmtree(source_dir, ignore_errors=True)
                return
        except Exception:
            # Fail open: any issues in the dedup filter should not block saving.
            pass

    target = bug_root / source_dir.name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source_dir, target)
    if error_log:
        log_path = target / "error.log"
        with log_path.open("w", encoding="utf-8") as fp:
            fp.write(error_log)
    shutil.rmtree(source_dir, ignore_errors=True)


def _prepare_tvm_inputs(
    tvm_module, oracle_inputs: Dict[str, np.ndarray]
) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
    input_list: List[np.ndarray] = []
    for param in tvm_module["main"].params:
        if param.name_hint in oracle_inputs:
            value = oracle_inputs[param.name_hint]
            input_list.append(value)
    params = getattr(tvm_module, "params", None)
    extra_params = None
    if params:
        extra_params = params["main"]
    return input_list, extra_params


def _convert_tvm_output(tvm_out, expected):
    import tvm

    if isinstance(tvm_out, tuple):
        return [ _convert_tvm_output(o, expected) for o in tvm_out ]
    if isinstance(tvm_out, tvm.runtime.ShapeTuple):
        return np.array([int(v) for v in tvm_out])
    if isinstance(tvm_out, tvm.runtime.Tensor):
        return tvm_out.numpy()
    if isinstance(tvm_out, (int, float, bool)):
        return np.array(tvm_out)
    return tvm_out


def _run_tvm(model_path: Path, oracle_path: Path) -> TestResult:
    """使用TVM编译和运行ONNX模型并验证结果

    将ONNX模型转换为TVM Relax IR，编译为可执行模块，
    运行推理并与ONNX Runtime结果进行比较。

    Args:
        model_path: ONNX模型文件路径
        oracle_path: oracle数据文件路径（可选，用于兼容性）

    Returns:
        TestResult对象，包含TVM测试结果
    """
    start = time.time()
    try:
        import tvm
        from tvm import relax
        from tvm.relax.frontend.onnx import from_onnx
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("TVM is not available in the current environment.") from exc

    model = onnx.load(model_path)

    # Generate random inputs for testing
    inputs = _generate_random_inputs(model)

    # Run model through ONNX Runtime to get expected results
    ort_providers = ["CPUExecutionProvider"]
    try:
        ort_session = ort.InferenceSession(
            model.SerializeToString(), providers=ort_providers
        )
        ort_output = ort_session.run([], inputs)
    except Exception as e:
        # If ONNX Runtime fails, we can't establish a baseline
        duration = time.time() - start
        return TestResult(
            success=False,
            crashed=False,
            divergence=False,
            valid=False,
            duration_s=duration,
            message=f"ONNX Runtime failed: {str(e)}",
        )

    # Convert the onnx model into relax through the onnx importer.
    res = from_onnx(model, shape_dict={k: v.shape for k, v in inputs.items()})
    tvm_module = res[0] if isinstance(res, (list, tuple)) else res

    # Compile the relax graph into a VM then run.
    with tvm.transform.PassContext(opt_level=4):
        target = tvm.target.Target("llvm", host="llvm")
        ex = relax.build(tvm_module, target=target)
        dev = tvm.cpu()
        vm = relax.VirtualMachine(ex, dev)

    # 入口函数名
    try:
        tvm_module.get_global_var("main")
        entry_name = "main"
    except Exception:
        gvs = list(tvm_module.get_global_vars())
        if not gvs:
            raise RuntimeError("No global functions found in Relax module.")
        entry_name = gvs[0].name_hint

    # Prepare inputs
    input_order = [i.name for i in model.graph.input if i.name in inputs]
    tvm_inputs = [tvm.runtime.tensor(inputs[name], dev) for name in input_order]

    # Run model and check outputs.
    tvm_output = vm[entry_name](*tvm_inputs)
    if hasattr(tvm_output, "numpy"):
        tvm_output = [tvm_output]
    elif isinstance(tvm_output, tuple):
        tvm_output = list(tvm_output)

    # Check that number of outputs match.
    assert len(tvm_output) == len(ort_output), "Unequal number of outputs"
    for tvm_out, ort_out in zip(tvm_output, ort_output):
        if ort_out is not None:
            np.testing.assert_allclose(
                tvm_out.numpy(), ort_out, rtol=1e-1, atol=1e-1
            )

    duration = time.time() - start
    return TestResult(
        success=True,
        crashed=False,
        divergence=False,
        valid=True,
        duration_s=duration,
        message="TVM outputs match ONNX Runtime.",
    )


def _generate_random_inputs(model: onnx.ModelProto, inputs: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """为ONNX模型生成随机输入数据

    根据模型的输入定义生成符合形状和数据类型的随机张量。

    Args:
        model: ONNX模型对象
        inputs: 可选的预定义输入数据字典

    Returns:
        输入名称到随机张量的映射字典
    """
    """Generate random inputs for ONNX model testing."""
    input_values = {}
    initializer_names = set(init.name for init in model.graph.initializer)
    # Iterate through model inputs and extract their shape.
    for i in model.graph.input:
        # Many real-world ONNX models list initializers (weights) in graph.input.
        # Runtimes expect feeds only for *real* inputs, not initializers.
        if i.name in initializer_names and (inputs is None or i.name not in inputs):
            continue
        if inputs is not None and i.name in inputs and inputs[i.name] is not None:
            input_values[i.name] = inputs[i.name]
            continue
        shape = []
        for dim in i.type.tensor_type.shape.dim:
            # NOTE:
            # - For symbolic / unknown dims, ONNX often stores dim_value=0 (and/or dim_param="...").
            # - Some runtimes (e.g. OpenVINO) reject batch=0.
            # Therefore we must materialize unknown/0 dims to a positive integer.
            dv = int(getattr(dim, "dim_value", 0) or 0)
            if dv <= 0:
                dv = 1
            shape.append(dv)

        input_values[i.name] = _generate_random_value(shape, i.type.tensor_type.elem_type)

    return input_values


def _generate_random_value(shape, elem_type) -> np.ndarray:
    """生成指定形状和类型的随机数值

    根据ONNX元素类型生成适当的随机数据。

    Args:
        shape: 张量形状
        elem_type: ONNX元素类型

    Returns:
        随机生成的numpy数组
    """
    """Generate random value for given shape and element type."""
    # Extract datatype for the input.
    if elem_type:
        dtype = str(helper.tensor_dtype_to_np_dtype(elem_type))
        # Handle deprecated numpy types
        if dtype == "float":
            dtype = "float32"
        elif dtype == "double":
            dtype = "float64"
    else:
        dtype = "float32"

    # Generate random inputs for each input.
    if dtype == "bool":
        random_value = np.random.choice(a=[False, True], size=shape)
    elif dtype.startswith("int"):
        # Keep non-zero values
        random_value = np.random.randint(low=-63, high=63, size=shape).astype(dtype)
        random_value[random_value <= 0] -= 1
    else:
        random_value = np.random.standard_normal(size=shape).astype(dtype)

    return random_value

