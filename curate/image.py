"""Image signals. Images are binary columns (encoded bytes)."""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa

from .signal import Signal, matrix, vectors

_pool = ThreadPoolExecutor(16)


def _pil(b: bytes):
    """Decode to RGB; a corrupt file becomes a grey image instead of killing the job."""
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        return Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), (128, 128, 128))


class _Clip(Signal):
    gpu = True
    batch_size = 512

    def __init__(self, *inputs, model="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        super().__init__(*inputs)
        self.model_name, self.pretrained = model, pretrained

    def setup(self):
        import open_clip
        import torch

        self.model, _, self.pre = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained, precision="fp16", device="cuda")
        self.model.eval()
        self.tok = open_clip.get_tokenizer(self.model_name)
        self.torch = torch

    def image_emb(self, images: pa.Array):
        px = list(_pool.map(lambda b: self.pre(_pil(b)), images.to_pylist()))
        x = self.torch.stack(px).cuda().half()
        with self.torch.inference_mode():
            return self.torch.nn.functional.normalize(self.model.encode_image(x).float(), dim=1)

    def text_emb(self, captions: pa.Array):
        toks = self.tok([c or "" for c in captions.to_pylist()]).cuda()
        with self.torch.inference_mode():
            return self.torch.nn.functional.normalize(self.model.encode_text(toks).float(), dim=1)


class embed(_Clip):
    """CLIP image embedding (ViT-B/32 by default, 512-d)."""

    def __init__(self, column, dim=512, **kw):
        super().__init__(column, **kw)
        self.dtype = pa.list_(pa.float32(), dim)

    def compute(self, image):
        return vectors(self.image_emb(image).cpu().numpy())


class clip_score(_Clip):
    """Cosine similarity between CLIP image and caption embeddings."""

    def compute(self, image, caption):
        return (self.image_emb(image) * self.text_emb(caption)).sum(1).cpu().numpy()


class aesthetic(Signal):
    """LAION aesthetic predictor (1..10) on a CLIP ViT-L/14 embedding column, or on
    raw images (then the L/14 model runs first)."""

    gpu = True
    batch_size = 512
    URL = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"

    def setup(self):
        import os
        import urllib.request

        import torch

        path = os.path.expanduser("~/.cache/aesthetic/sac+logos+ava1-l14-linearMSE.pth")
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve(self.URL, path)
        n = torch.nn
        self.head = n.Sequential(n.Linear(768, 1024), n.Dropout(0.2), n.Linear(1024, 128), n.Dropout(0.2), n.Linear(128, 64), n.Dropout(0.1), n.Linear(64, 16), n.Linear(16, 1))
        sd = torch.load(path, map_location="cpu")
        self.head.load_state_dict({k.removeprefix("layers."): v for k, v in sd.items()})
        self.head.cuda().eval()
        self.clip = None

    def compute(self, x):
        import torch

        if pa.types.is_binary(x.type) or pa.types.is_large_binary(x.type):
            if self.clip is None:
                self.clip = embed(self.inputs[0], dim=768, model="ViT-L-14", pretrained="openai")
                self.clip.setup()
            e = self.clip.image_emb(x)
        else:
            e = torch.nn.functional.normalize(torch.from_numpy(matrix(x)).cuda().float(), dim=1)
        with torch.inference_mode():
            return self.head(e).squeeze(1).cpu().numpy()


class resolution(Signal):
    """Shorter side in pixels, from (width, height) columns or from image bytes."""

    dtype = "int32"
    batch_size = 4096

    def compute(self, *cols):
        if len(cols) == 2:
            return np.minimum(cols[0].to_numpy(zero_copy_only=False), cols[1].to_numpy(zero_copy_only=False)).astype(np.int32)
        from PIL import Image

        return [min(Image.open(io.BytesIO(b)).size) for b in cols[0].to_pylist()]
