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
