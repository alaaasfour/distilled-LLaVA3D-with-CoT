#!/usr/bin/env python3
"""Utility to load the LLaVA-3D teacher model inside the distillation workspace."""

from __future__ import annotations

import importlib.machinery as importlib_machinery
import math
import os
import sys
import types
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple

import torch

LLAVA3D_REPO = Path("/home/alasfour/scratch/llava-3d/LLaVA-3D")
LLAVA3D_PACKAGE = LLAVA3D_REPO / "llava"

if not LLAVA3D_PACKAGE.is_dir():
    raise FileNotFoundError(
        "Expected the LLaVA-3D python package under "
        f"{LLAVA3D_PACKAGE}. Please verify the repository path."
    )

# Make sure the LLaVA-3D package is importable before we touch any llava.* modules.
repo_path = str(LLAVA3D_REPO)
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

package_path = str(LLAVA3D_PACKAGE)
if package_path not in sys.path:
    sys.path.insert(0, package_path)


try:
    import cv2  # type: ignore
except ModuleNotFoundError:
    warnings.warn(
        "OpenCV (cv2) is not installed; depth inpainting will be skipped. "
        "Install opencv-python-headless for the full feature set."
    )

    def _cv2_inpaint(depth, mask, inpaint_radius, flags=None):
        return depth

    cv2_stub = types.ModuleType("cv2_stub")
    cv2_stub.__spec__ = importlib_machinery.ModuleSpec("cv2", loader=None)  # type: ignore[attr-defined]
    cv2_stub.inpaint = _cv2_inpaint  # type: ignore[attr-defined]
    cv2_stub.INPAINT_NS = 0  # type: ignore[attr-defined]
    sys.modules["cv2"] = cv2_stub

try:
    from scipy.spatial.transform import Rotation as R  # type: ignore
except ModuleNotFoundError:
    warnings.warn(
        "SciPy is not installed; using an approximate quaternion-to-matrix conversion. "
        "Install scipy for higher accuracy."
    )

    import numpy as _np

    class _Rotation:
        def __init__(self, matrix: _np.ndarray):
            self._matrix = matrix

        @classmethod
        def from_quat(cls, quat):
            x, y, z, w = (float(q) for q in quat)
            norm = math.sqrt(x * x + y * y + z * z + w * w)
            if norm == 0:
                raise ValueError("Zero-norm quaternion is invalid")
            x /= norm
            y /= norm
            z /= norm
            w /= norm
            matrix = _np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ], dtype=float)
            return cls(matrix)

        def as_matrix(self):
            return self._matrix

    scipy_module = types.ModuleType("scipy")
    scipy_module.__spec__ = importlib_machinery.ModuleSpec("scipy", loader=None)  # type: ignore[attr-defined]

    spatial_module = types.ModuleType("scipy.spatial")
    spatial_module.__spec__ = importlib_machinery.ModuleSpec("scipy.spatial", loader=None)  # type: ignore[attr-defined]

    transform_module = types.ModuleType("scipy.spatial.transform")
    transform_module.__spec__ = importlib_machinery.ModuleSpec("scipy.spatial.transform", loader=None)  # type: ignore[attr-defined]
    transform_module.Rotation = _Rotation  # type: ignore[attr-defined]

    spatial_module.transform = transform_module  # type: ignore[attr-defined]
    scipy_module.spatial = spatial_module  # type: ignore[attr-defined]

    sys.modules.setdefault("scipy", scipy_module)
    sys.modules.setdefault("scipy.spatial", spatial_module)
    sys.modules.setdefault("scipy.spatial.transform", transform_module)
    R = _Rotation


@contextmanager
def _temporary_cwd(path: Path):
    original_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original_cwd)

with _temporary_cwd(LLAVA3D_REPO):
    from llava.model.builder import load_pretrained_model  # noqa: E402
    from llava.mm_utils import get_model_name_from_path  # noqa: E402


def load_llava3d_teacher(
    model_path: str = "ChaimZhu/LLaVA-3D-7B",
    device: str = "cuda",
    precision: str | torch.dtype = "bf16",
    quant: str | None = None,
) -> Tuple[object, object, object, int]:
    """Load the LLaVA-3D teacher model and return the HF components."""

    if device.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    if isinstance(precision, str):
        dtype_aliases = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        try:
            precision = dtype_aliases[precision.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported precision '{precision}'.") from exc

    if device == "cpu" and precision in {torch.float16, torch.bfloat16}:
        warnings.warn("Half precision on CPU is unstable. Using float32 instead.")
        precision = torch.float32

    if quant in {"4bit", "8bit"} and device != "cuda":
        warnings.warn("Quantization requires CUDA. Disabling quantization.")
        quant = None

    load_4bit = quant == "4bit"
    load_8bit = quant == "8bit"

    # Avoid passing `device_map="auto"` since CLIPVisionModel does not
    # implement `_no_split_modules` and Accelerate will raise.
    device_map = None

    try:
        with _temporary_cwd(LLAVA3D_REPO):
            tokenizer, model, processor, context_len = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=get_model_name_from_path(model_path),
                load_8bit=load_8bit,
                load_4bit=load_4bit,
                device_map=device_map,
                device=device,
                torch_dtype=precision,
            )
    except Exception as exc:  # noqa: BLE001
        error_msg = str(exc).lower()
        if quant in {"4bit", "8bit"} and "bitsandbytes" in error_msg:
            warnings.warn(
                "bitsandbytes quantization failed; retrying without quantization."
            )
            return load_llava3d_teacher(
                model_path=model_path,
                device=device,
                precision=precision,
                quant=None,
            )
        raise

    return tokenizer, model, processor, context_len


if __name__ == "__main__":
    print("Loading LLaVA-3D teacher model...")
    tokenizer, model, processor, context_len = load_llava3d_teacher()
    print("Teacher model loaded successfully!")
    print(f"Context length: {context_len}")
