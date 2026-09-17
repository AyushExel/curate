"""Text signals for LLM pretraining / post-training data."""

from __future__ import annotations

import hashlib

import pyarrow as pa
import pyarrow.compute as pc

from .signal import Signal, vectors


class length(Signal):
    """Number of whitespace-separated words."""

    dtype = "int32"

    def compute(self, text):
        return pc.cast(pc.count_substring_regex(text, r"\S+"), pa.int32())


class language(Signal):
    """fastText language id (e.g. ``eng_Latn``); ``und`` below ``min_confidence``."""

    dtype = "string"
    batch_size = 4096

    def __init__(self, column, min_confidence=0.5):
        super().__init__(column)
        self.min_confidence = min_confidence

    def setup(self):
        import fasttext
        from huggingface_hub import hf_hub_download

        self.model = fasttext.load_model(hf_hub_download("facebook/fasttext-language-identification", "model.bin"))

    def compute(self, text):
        docs = [" ".join(t.split())[:2000] for t in text.to_pylist()]
        labels, probs = self.model.predict(docs)
        return [l[0].removeprefix("__label__") if p[0] >= self.min_confidence else "und" for l, p in zip(labels, probs)]


class quality(Signal):
    """HuggingFaceFW/fineweb-edu-classifier score, roughly 0..5 (>=3 is 'educational')."""

    gpu = True
    batch_size = 256

    def __init__(self, column, model="HuggingFaceFW/fineweb-edu-classifier", max_length=512):
        super().__init__(column)
        self.model_name, self.max_length = model, max_length

    def setup(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, torch_dtype=torch.float16).cuda().eval()

    def compute(self, text):
        import torch

        enc = self.tok(text.to_pylist(), padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            return self.model(**enc).logits.squeeze(-1).float().cpu().numpy()


class embed(Signal):
    """Sentence embedding (MiniLM-L6 by default, 384-d, L2-normalized)."""

    gpu = True
    batch_size = 1024

    def __init__(self, column, model="sentence-transformers/all-MiniLM-L6-v2", dim=384):
        super().__init__(column)
        self.model_name, self.dim = model, dim
        self.dtype = pa.list_(pa.float32(), dim)

    def setup(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_name, device="cuda", model_kwargs={"torch_dtype": "float16"})

    def compute(self, text):
        return vectors(self.model.encode(text.to_pylist(), batch_size=256, normalize_embeddings=True, show_progress_bar=False))


class tokenize(Signal):
    """Token ids from a HuggingFace tokenizer."""

    dtype = pa.list_(pa.int32())
    batch_size = 1024

    def __init__(self, column, tokenizer="gpt2", max_length=None):
        super().__init__(column)
        self.tokenizer, self.max_length = tokenizer, max_length

    def setup(self):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(self.tokenizer)

    def compute(self, text):
        kw = {"truncation": True, "max_length": self.max_length} if self.max_length else {}
        return pa.array(self.tok(text.to_pylist(), **kw)["input_ids"], pa.list_(pa.int32()))


class contaminated(Signal):
    """True if any word n-gram of the row appears in ``against`` (an eval set)."""

    dtype = "bool"
    batch_size = 4096

    def __init__(self, column, against, ngram=13):
        super().__init__(column)
        self.ngram = ngram
        self._against = list(against)
        self.n_against = len(self._against)

    def setup(self):
        self.bank = set()
        for doc in self._against:
            self.bank.update(_ngrams(doc, self.ngram))

    def compute(self, text):
        return [any(g in self.bank for g in _ngrams(t, self.ngram)) for t in text.to_pylist()]


def _ngrams(text: str, n: int):
    w = text.lower().split()
    if len(w) < n:
        return [hashlib.md5(" ".join(w).encode()).digest()] if w else []
    return [hashlib.md5(" ".join(w[i : i + n]).encode()).digest() for i in range(len(w) - n + 1)]
