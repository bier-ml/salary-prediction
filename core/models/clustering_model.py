from typing import Dict, Any, Self

import numpy as np
from sklearn.cluster import KMeans

from core.models.base_model import BaseMatchingModel


class ClusteringModel(BaseMatchingModel):
    def __init__(self, model_name: str = "kmeans", n_clusters: int = 17):
        available_models: Dict[str, Any] = {
            "kmeans": KMeans,
        }

        self.model = available_models[model_name](n_clusters=n_clusters)

    def train(self, embeddings: np.ndarray) -> Self:
        self.model.fit(embeddings)
        return self

    def predict(self, embedding: np.ndarray) -> int:
        return self.model.predict(embedding)[0]
