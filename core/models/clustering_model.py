from typing import Dict, Any, Self

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error

from core.models import CatboostRegressionModel
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


class StackedModels(BaseMatchingModel):
    def __init__(self, base_model_class: type[CatboostRegressionModel] = CatboostRegressionModel):
        self.base_model_class = base_model_class
        self.models: Dict[int, BaseMatchingModel] = {}

    def train(self, dataset_dict: Dict[int, pd.DataFrame], **train_kwargs) -> Self:
        for cluster_label, dataset in dataset_dict.items():
            model = self.base_model_class()
            model.train(dataset=dataset, **train_kwargs)
            self.models[cluster_label] = model

        print("Model trained")

        # test_score = self.evaluate(...)
        # print(f"Test score is {test_score}")
        return self

    def evaluate(self, X_dict: Dict[int, Any], y_dict: Dict[int, Any]) -> float:
        return sum(
            mean_absolute_error(y_dict[cluster], self.models[cluster].predict(X_dict[cluster]))
            for cluster in self.models
        ) / len(self.models)

    def predict(self, embedding: np.ndarray, cluster_label: int = 0) -> float:
        return self.models[cluster_label].predict(embedding)
