"""Video signals. Clips are binary/blob columns holding an encoded container."""

from __future__ import annotations

import io

import numpy as np
import pyarrow as pa

from .image import _Clip, _pool
from .signal import Signal


def frames(blob: bytes, n=16, size=64) -> np.ndarray:
    """Decode ``n`` evenly spaced frames, resized to ``size``x``size`` RGB uint8."""
    import av
    from PIL import Image

    with av.open(io.BytesIO(blob)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        total = s.frames or int((c.duration or 0) / 1e6 * float(s.average_rate or 24)) or n
        want = set(np.linspace(0, max(total - 1, 0), n).astype(int).tolist())
        out = []
        for i, f in enumerate(c.decode(s)):
            if i in want:
                out.append(np.asarray(Image.fromarray(f.to_ndarray(format="rgb24")).resize((size, size))))
            if len(out) == len(want):
                break
    return np.stack(out) if out else np.zeros((1, size, size, 3), np.uint8)


class motion(Signal):
    """Mean absolute grayscale difference between consecutive sampled frames (0..1)."""

    batch_size = 16

    def __init__(self, column, n_frames=16):
        super().__init__(column)
        self.n_frames = n_frames

    def one(self, b):
        g = frames(b, self.n_frames).mean(-1) / 255.0
        return float(np.abs(np.diff(g, axis=0)).mean()) if len(g) > 1 else 0.0

    def compute(self, video):
        return list(_pool.map(self.one, video.to_pylist()))


class scene_cuts(Signal):
    """Number of hard cuts: colour-histogram L1 distance between consecutive frames above ``threshold``."""

    dtype = "int32"
    batch_size = 16

    def __init__(self, column, n_frames=32, threshold=0.5):
        super().__init__(column)
        self.n_frames, self.threshold = n_frames, threshold

    def one(self, b):
        fr = frames(b, self.n_frames)
        h = np.stack([np.concatenate([np.histogram(f[..., c], 16, (0, 255))[0] for c in range(3)]) for f in fr]).astype(float)
        h /= h.sum(1, keepdims=True)
        return int((np.abs(np.diff(h, axis=0)).sum(1) > self.threshold).sum())

    def compute(self, video):
        return list(_pool.map(self.one, video.to_pylist()))


class clip_score(_Clip):
    """CLIP similarity between the caption and the mean of ``n_frames`` frame embeddings."""

    batch_size = 32

    def __init__(self, video, caption, n_frames=4, **kw):
        super().__init__(video, caption, **kw)
        self.n_frames = n_frames

    def compute(self, video, caption):
        import torch

        clips = list(_pool.map(lambda b: frames(b, self.n_frames, size=224), video.to_pylist()))
        px = [self.pre(_to_pil(f)) for fr in clips for f in fr]
        x = torch.stack(px).cuda().half()
        with torch.inference_mode():
            e = torch.nn.functional.normalize(self.model.encode_image(x).float(), dim=1)
        e = torch.nn.functional.normalize(e.view(len(video), -1, e.shape[1]).mean(1), dim=1)
        return (e * self.text_emb(caption)).sum(1).cpu().numpy()


def _to_pil(arr):
    from PIL import Image

    return Image.fromarray(arr)
