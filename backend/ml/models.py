"""Lightweight ML helpers for Spectra admin telemetry.

These models intentionally delay importing torch/sklearn until runtime so the
public /explore traffic does not incur a cold-start cost on boot.
"""

from __future__ import annotations

from typing import Any


class DipoleConvNet3D:
    """Small 3D convnet stub used for telemetry and demo replay."""

    def __init__(self, name: str = "DipoleConvNet3D") -> None:
        self.name = name

    def forward(self, x: Any) -> Any:
        backend = self._require_torch()
        if backend is None:
            return {"model": self.name, "shape": getattr(x, "shape", (0, 0, 0, 0, 0)), "status": "demo"}
        tensor = backend.as_tensor(x) if not isinstance(x, backend.Tensor) else x
        return tensor

    def predict(self, x: Any) -> Any:
        return self.forward(x)

    @staticmethod
    def _require_torch() -> Any | None:
        try:
            import torch  # type: ignore
        except ModuleNotFoundError:
            return None
        return torch


class SpectralResNet102:
    """Lightweight spectral residual model stub."""

    def __init__(self, name: str = "SpectralResNet102") -> None:
        self.name = name

    def forward(self, x: Any) -> Any:
        backend = self._require_torch()
        if backend is None:
            return {"model": self.name, "shape": getattr(x, "shape", (0, 0)), "status": "demo"}
        tensor = backend.as_tensor(x) if not isinstance(x, backend.Tensor) else x
        return tensor

    def predict(self, x: Any) -> Any:
        return self.forward(x)

    @staticmethod
    def _require_torch() -> Any | None:
        try:
            import torch  # type: ignore
        except ModuleNotFoundError:
            return None
        return torch


class AstroFeatureSVM:
    """Feature-space SVM stub with no eager sklearn import."""

    def __init__(self, name: str = "AstroFeatureSVM") -> None:
        self.name = name

    def fit(self, X: Any, y: Any) -> "AstroFeatureSVM":
        return self

    def predict(self, X: Any) -> Any:
        try:
            import sklearn  # noqa: F401  # type: ignore
        except ModuleNotFoundError:
            return {"model": self.name, "status": "demo", "predictions": []}
        return {"model": self.name, "status": "ready", "predictions": []}


__all__ = ["DipoleConvNet3D", "SpectralResNet102", "AstroFeatureSVM"]
