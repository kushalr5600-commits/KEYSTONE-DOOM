"""KEYSTONE-DOOM: Key-Event Structured Topological Memory.

Compact, learned, contract-compatible policy modules shared by training and inference.

Design boundary (docs/architecture.md): every input here is derived from the official
`doom_env` observation dictionary only. No privileged simulator state, no map data, no
engine handle, no network access. All decisions are produced by trained network weights.
"""

from .config import KeyStoneConfig
from .features import FEATURE_DIM, EVIDENCE_DIM, FeatureBuilder
from .memory import EventMemory, MemoryState
from .model import KeyStoneNet, AgentState
from .actions import MACRO_ACTIONS, MACRO_NAMES, legalize_action, macro_action

__all__ = [
    "KeyStoneConfig",
    "FeatureBuilder",
    "FEATURE_DIM",
    "EVIDENCE_DIM",
    "EventMemory",
    "MemoryState",
    "KeyStoneNet",
    "AgentState",
    "legalize_action", "macro_action", "MACRO_ACTIONS", "MACRO_NAMES",
]
