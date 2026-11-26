"""Centralized device selection for training (deployment always uses the contract device).

CPU fallback remains the default when CUDA is unavailable. Models, batches, optimizers and
losses are moved consistently by callers. This module never affects inference in
submission/agent.py, which uses the device argument supplied by the scorer.
"""

from __future__ import annotations

import argparse
import torch


def get_device(preference: str = "auto") -> torch.device:
    """Resolve 'auto'|'cuda'|'cpu' into an available torch.device."""
    pref = str(preference).lower()
    if pref == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if pref in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def add_device_arg(parser: argparse.ArgumentParser, default: str = "auto") -> None:
    parser.add_argument("--device", default=default, choices=["auto", "cuda", "cpu"],
                        help="training device (deployment device is supplied by the scorer)")


def describe(device: torch.device) -> str:
    if device.type == "cuda":
        return (f"cuda:{torch.cuda.current_device()} "
                f"{torch.cuda.get_device_name(device)} "
                f"{torch.cuda.get_device_properties(device).total_memory / 1024**2:.0f} MB")
    return "cpu"
