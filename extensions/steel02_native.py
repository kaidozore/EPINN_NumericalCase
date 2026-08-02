"""Compiled CPU Steel02 history kernel; CUDA builds use the same public API."""

from __future__ import annotations

import ctypes
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

import torch

_LIBRARY = None
_FAILED = False
_CUDA_MODULE = None
_CUDA_FAILED = False


def _configure_cuda_architecture() -> None:
    """Use forward-compatible PTX when an old NVCC sees an Ada GPU."""

    if os.environ.get("TORCH_CUDA_ARCH_LIST") or not torch.cuda.is_available():
        return
    capability = torch.cuda.get_device_capability()
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return
    result = subprocess.run(
        [nvcc, "--version"], capture_output=True, text=True, check=False
    )
    match = re.search(r"release\s+(\d+)\.(\d+)", result.stdout + result.stderr)
    if match is None:
        return
    nvcc_version = (int(match.group(1)), int(match.group(2)))
    if capability >= (8, 9) and nvcc_version < (11, 8):
        # CUDA before 11.8 cannot emit sm_89.  Compute-86 PTX is JIT-compiled
        # by the current NVIDIA driver for Ada while preserving float64 math.
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6+PTX"
        print(
            "Steel02 CUDA compatibility: GPU sm_"
            f"{capability[0]}{capability[1]}, NVCC {nvcc_version[0]}."
            f"{nvcc_version[1]}; compiling 8.6+PTX."
        )


def _load_cpu_library():
    global _LIBRARY, _FAILED
    if _LIBRARY is not None or _FAILED:
        return _LIBRARY
    root = Path(__file__).resolve().parent
    build = root / "_build"
    suffix = ".dll" if os.name == "nt" else ".so"
    library = build / f"steel02_native{suffix}"
    source = root / "steel02_native.cpp"
    try:
        if not library.exists() or library.stat().st_mtime < source.stat().st_mtime:
            build.mkdir(parents=True, exist_ok=True)
            compiler = shutil.which("g++")
            if compiler is None:
                raise RuntimeError("g++ was not found")
            command = [compiler, "-O3", "-std=c++17", "-shared", "-fopenmp"]
            if os.name != "nt":
                command.append("-fPIC")
            command += [str(source), "-o", str(library)]
            subprocess.run(command, check=True, capture_output=True, text=True)
        if os.name == "nt":
            compiler = shutil.which("g++")
            compiler_directory = Path(compiler).parent
            for name in ("libgomp-1.dll", "libwinpthread-1.dll", "libgcc_s_seh-1.dll"):
                dependency = compiler_directory / name
                if dependency.exists():
                    shutil.copy2(dependency, build / dependency.name)
        _LIBRARY = ctypes.CDLL(str(library))
        _LIBRARY.steel02_forward.restype = None
    except Exception as error:
        _FAILED = True
        print(f"Steel02 native CPU kernel unavailable; using PyTorch fallback: {error}")
    return _LIBRARY


def available(device: torch.device) -> bool:
    return (
        _load_cpu_library() is not None
        if device.type == "cpu"
        else _load_cuda_module() is not None
    )


def _load_cuda_module():
    global _CUDA_MODULE, _CUDA_FAILED
    if _CUDA_MODULE is not None or _CUDA_FAILED:
        return _CUDA_MODULE
    try:
        _configure_cuda_architecture()
        from torch.utils.cpp_extension import load
        root = Path(__file__).resolve().parent
        build = root / "_build_cuda"
        build.mkdir(parents=True, exist_ok=True)
        _CUDA_MODULE = load(
            name="steel02_cuda_ext",
            sources=[str(root / "steel02_cuda.cpp"), str(root / "steel02_cuda_kernel.cu")],
            build_directory=str(build),
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
        print("Steel02 CUDA extension loaded successfully.")
    except Exception as error:
        _CUDA_FAILED = True
        print(f"Steel02 CUDA kernel unavailable; using PyTorch fallback: {error}")
    return _CUDA_MODULE


def forward(displacement, curvature, force_map, fiber_y, fiber_area, params, state):
    if displacement.device.type == "cuda":
        module = _load_cuda_module()
        if module is None:
            raise RuntimeError("Steel02 CUDA kernel is unavailable")
        return module.forward(
            displacement, curvature, force_map, fiber_y, fiber_area, params, state
        )
    library = _load_cpu_library()
    if library is None:
        raise RuntimeError("Steel02 native kernel is unavailable")
    tensors = [displacement, curvature, force_map, fiber_y, fiber_area, params, state]
    if any(value.device.type != "cpu" or value.dtype != torch.float64 for value in tensors):
        raise TypeError("Native CPU Steel02 requires contiguous CPU float64 tensors")
    tensors = [value.contiguous() for value in tensors]
    displacement, curvature, force_map, fiber_y, fiber_area, params, state = tensors
    batch, steps, reduced = displacement.shape
    elements, gauss, _ = curvature.shape
    fibers = fiber_y.numel()
    internal = displacement.new_zeros(batch, steps, reduced)
    tangent = displacement.new_zeros(batch, steps, reduced, reduced)
    final_state = torch.empty_like(state)
    pointer = ctypes.c_void_p
    library.steel02_forward(
        pointer(displacement.data_ptr()), batch, steps, reduced,
        pointer(curvature.data_ptr()), pointer(force_map.data_ptr()),
        elements, gauss, pointer(fiber_y.data_ptr()), pointer(fiber_area.data_ptr()),
        fibers, pointer(params.data_ptr()), pointer(state.data_ptr()),
        pointer(internal.data_ptr()), pointer(tangent.data_ptr()),
        pointer(final_state.data_ptr()),
    )
    return internal, tangent, final_state
