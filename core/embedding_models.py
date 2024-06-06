from typing import Iterable

import numpy as np
from gensim.models.fasttext import FastText
from sentence_transformers import SentenceTransformer, util

from core import ROOT_PATH

AVAILABLE_EMBEDDINGS: set[str] = {
    "cointegrated/rubert-tiny2",
    "DeepPavlov/rubert-base-cased",
    "DeepPavlov/bert-base-bg-cs-pl-ru-cased",
    "DeepPavlov/rubert-base-cased-sentence",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
    "cointegrated/LaBSE-en-ru",
}


class DummyEmbeddingModel:
    def __init__(self):
        np.random.seed(42)

    @staticmethod
    def generate(text: str | list[str], dim: int = 312) -> np.ndarray | Iterable[np.ndarray]:
        if isinstance(text, str):
            return np.random.rand(dim)
        return np.random.rand(len(text), dim)


class EmbeddingModel:
    def __init__(self, model_name: str = "cointegrated/LaBSE-en-ru"):
        self.model = SentenceTransformer(model_name)

    def generate(self, text: str | Iterable[str]) -> np.ndarray:
        return self.model.encode(text)


class FastTextEmbeddingModel:
    def __init__(self, model_source: str = str(ROOT_PATH / "data/cc.ru.300.bin")):
        self.model = FastText.load_fasttext_format(model_source)

    @staticmethod
    def get_sentence_vector(model: FastText, s: str) -> np.ndarray:
        return np.mean([model.wv[s_i] for s_i in s.split()], axis=0)

    def generate(self, text: str | Iterable[str]) -> np.ndarray:
        return self.get_sentence_vector(self.model, text)


if __name__ == "__main__":
    embedding_model = EmbeddingModel()
    v1, v2, v3 = (
        embedding_model.generate("водитель"),
        embedding_model.generate("пилот"),
        embedding_model.generate("художник"),
    )
    print(util.pytorch_cos_sim(v1, v2))
    print(util.pytorch_cos_sim(v1, v3))
