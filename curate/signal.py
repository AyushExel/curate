"""A Signal is a function of columns that produces one new column.

Built-in signals are classes; you bind one to columns by constructing it
(``text.quality("text")``).  Anything that needs a model loads it in
``setup()``, which runs once per worker.  ``compute`` gets one pyarrow Array
per input column and returns the new column for that batch.
"""

from __future__ import annotations

import inspect

import numpy as np
import pyarrow as pa


def dtype(t) -> pa.DataType:
    """Accept a pyarrow type or a short alias like ``"float32"`` / ``"bool"``."""
    return t if isinstance(t, pa.DataType) else pa.type_for_alias(t)


def vectors(x: np.ndarray) -> pa.FixedSizeListArray:
    """2-D float array -> fixed-size-list<float32> column."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(x.ravel()), x.shape[1])


def matrix(arr: pa.Array) -> np.ndarray:
    """fixed-size-list column (or slice of one) -> 2-D numpy array."""
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    return arr.flatten().to_numpy(zero_copy_only=False).reshape(len(arr), -1)


class Signal:
    dtype = "float32"
    gpu = False
    batch_size = 1024

    def __init__(self, *inputs: str):
        self.inputs = list(inputs)
        self._ready = False

    @property
    def out_type(self) -> pa.DataType:
        return dtype(self.dtype)

    def setup(self) -> None:
        """Load models here. Runs once per worker, lazily."""

    def compute(self, *arrays: pa.Array):
        raise NotImplementedError

    def __call__(self, *arrays: pa.Array) -> pa.Array:
        """Engines call this. Chunks by ``batch_size`` so compute() never sees more
        rows than it asked for, whatever the engine's IO batch is."""
        if not self._ready:
            self.setup()
            self._ready = True
        n = len(arrays[0])
        parts = [self._one(*[a.slice(lo, self.batch_size) for a in arrays]) for lo in range(0, n, self.batch_size)]
        return pa.concat_arrays(parts) if len(parts) > 1 else parts[0]

    def _one(self, *arrays: pa.Array) -> pa.Array:
        out = self.compute(*arrays)
        if isinstance(out, pa.ChunkedArray):
            out = out.combine_chunks()
        return out.cast(self.out_type) if isinstance(out, pa.Array) else pa.array(out, self.out_type)

    def describe(self) -> dict:
        params = {
            k: v
            for k, v in vars(self).items()
            if not k.startswith("_") and k != "inputs" and isinstance(v, (str, int, float, bool))
        }
        return {"signal": type(self).__name__, "inputs": self.inputs, **params}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(map(repr, self.inputs))})"


class GroupSignal(Signal):
    """Per-group signal: ``compute`` sees one whole group (an episode, a clip)
    and returns a scalar, which is broadcast to every row of the group."""

    per_group = True


def signal(dtype="float32", gpu=False, batch_size=1024, group=False):
    """Turn a plain function of arrays into a Signal class.

    >>> @signal(dtype="int32")
    ... def n_chars(text): return pc.utf8_length(text)
    >>> ds.signal(n_chars=n_chars("text"))      # or n_chars() to use the arg name
    """

    def deco(fn):
        params = list(inspect.signature(fn).parameters)
        base = GroupSignal if group else Signal

        def __init__(self, *inputs):
            base.__init__(self, *(inputs or params))

        return type(
            fn.__name__,
            (base,),
            {
                "dtype": dtype,
                "gpu": gpu,
                "batch_size": batch_size,
                "compute": staticmethod(fn),
                "__init__": __init__,
                "__doc__": fn.__doc__,
            },
        )

    return deco
