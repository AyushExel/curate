"""Per-episode signals for robot and world-model frame tables.

Use through ``ds.group("episode_index", order="frame_index").signal(...)``.
Each signal sees one episode's rows (in ``order``) and returns one scalar.
"""

from __future__ import annotations

import numpy as np

from .signal import GroupSignal, matrix


class length(GroupSignal):
    """Frames in the episode."""

    dtype = "int32"

    def compute(self, key):
        return len(key)


class jerk(GroupSignal):
    """Mean L2 norm of the second difference of the action: high = shaky teleop."""

    def compute(self, action):
        a = matrix(action)
        return float(np.linalg.norm(np.diff(a, n=2, axis=0), axis=1).mean()) if len(a) > 2 else 0.0


class idle_fraction(GroupSignal):
    """Fraction of steps where the action barely changes (||delta|| < eps)."""

    def __init__(self, action, eps=1e-3):
        super().__init__(action)
        self.eps = eps

    def compute(self, action):
        a = matrix(action)
        return float((np.linalg.norm(np.diff(a, axis=0), axis=1) < self.eps).mean()) if len(a) > 1 else 1.0


class path_length(GroupSignal):
    """Total distance travelled in state space."""

    def compute(self, state):
        s = matrix(state)
        return float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum()) if len(s) > 1 else 0.0


class success(GroupSignal):
    """True if the flag column is true anywhere in the episode."""

    dtype = "bool"

    def compute(self, flag):
        return bool(np.any(flag.to_numpy(zero_copy_only=False)))


class peak(GroupSignal):
    """Maximum of a per-frame column over the episode (e.g. best reward reached)."""

    def compute(self, values):
        return float(np.max(values.to_numpy(zero_copy_only=False)))
