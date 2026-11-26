"""Contract-level action validity.

This module does not decide *what* to do -- it only enforces the contract's button
semantics (docs/CONTRACT.md §3). Policies choose logits; argmax or sampling produces a
binary vector; this function removes physically meaningless combinations and re-presses
of edge-triggered buttons. It is applied identically during training and inference.
"""

from __future__ import annotations

import numpy as np

# Button indices from env/doom_env/spaces.py:57-68
FWD, BACK, STRAFE_L, STRAFE_R, TURN_L, TURN_R, ATTACK, USE, NEXT_WEAPON = range(9)


def _bits(*indices):
    a = np.zeros(9, dtype=np.int8)
    for i in indices:
        a[i] = 1
    return tuple(int(v) for v in a)


# Legal, useful button combinations. Learned categorical over these instead of nine
# independent Bernoullis: no contradictory pairs, far more sample-efficient on 1 CPU.
MACRO_ACTIONS = (
    _bits(),                                      # 0 stand / observe
    _bits(FWD),                                   # 1 forward
    _bits(FWD, TURN_L),                           # 2 forward + turn left
    _bits(FWD, TURN_R),                           # 3 forward + turn right
    _bits(TURN_L),                                # 4 turn left
    _bits(TURN_R),                                # 5 turn right
    _bits(STRAFE_L),                              # 6 strafe left
    _bits(STRAFE_R),                              # 7 strafe right
    _bits(ATTACK),                                # 8 attack
    _bits(FWD, ATTACK),                           # 9 forward + attack
    _bits(ATTACK, TURN_L),                        # 10 attack + turn left
    _bits(ATTACK, TURN_R),                        # 11 attack + turn right
    _bits(USE),                                   # 12 use
    _bits(FWD, USE),                              # 13 forward + use
    _bits(USE, TURN_L),                           # 14 use + turn left
    _bits(USE, TURN_R),                           # 15 use + turn right
    _bits(NEXT_WEAPON),                           # 16 cycle weapon
)
MACRO_NAMES = (
    "stand", "forward", "fwd_left", "fwd_right", "turn_left", "turn_right",
    "strafe_left", "strafe_right", "attack", "fwd_attack", "attack_left",
    "attack_right", "use", "fwd_use", "use_left", "use_right", "next_weapon",
)
MACRO_BITS = {tuple(v): i for i, v in enumerate(MACRO_ACTIONS)}


def macro_action(index: int) -> np.ndarray:
    """Return the 9-bit action for a macro index (clamped for safety)."""
    idx = int(index) % len(MACRO_ACTIONS)
    return np.asarray(MACRO_ACTIONS[idx], dtype=np.int8)


def legalize_action(action, prev_action=None) -> np.ndarray:
    a = np.asarray(action).astype(np.int8).ravel().copy()
    if a.shape != (9,):
        raise ValueError(f"action must have 9 buttons, got {a.shape}")
    # Mutually exclusive movement directions.
    if a[FWD] and a[BACK]:
        a[BACK] = 0
    if a[STRAFE_L] and a[STRAFE_R]:
        a[STRAFE_R] = 0
    if a[TURN_L] and a[TURN_R]:
        a[TURN_R] = 0
    # Vanilla ViZDoom fires USE / weapon-select on the button-down edge only.
    if prev_action is not None:
        p = np.asarray(prev_action).astype(np.int8).ravel()
        if p.size == 9:
            if a[USE] and p[USE]:
                a[USE] = 0
            if a[NEXT_WEAPON] and p[NEXT_WEAPON]:
                a[NEXT_WEAPON] = 0
    a[a != 0] = 1
    return a
