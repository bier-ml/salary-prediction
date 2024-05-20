from typing import Dict, Any, Self, Optional

import numpy as np
import optuna
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

    @staticmethod
    def retrieve_cluster_data(full_dataset: pd.DataFrame, cluster_label: int) -> pd.DataFrame:
        if "cluster_label" not in full_dataset.columns:
            raise KeyError("cluster_label not found in the DataFrame")

        return full_dataset[full_dataset["cluster_label"] == cluster_label]

    def train(self, dataset_dict: Dict[int, pd.DataFrame], **train_kwargs) -> Self:
        for cluster_label, dataset in dataset_dict.items():
            model = self.base_model_class()
            model.train(dataset=dataset, **train_kwargs)
            self.models[cluster_label] = model

        print("Model trained")
        return self

    def evaluate(self, X_dict: Dict[int, Any], y_dict: Dict[int, Any]) -> float:
        return sum(
            mean_absolute_error(y_dict[cluster], self.models[cluster].predict(X_dict[cluster]))
            for cluster in self.models
        ) / len(self.models)

    def predict(self, embedding: np.ndarray, cluster_label: int = 0) -> float:
        return self.models[cluster_label].predict(embedding)


def objective(trial, dataset_dict: Dict[int, pd.DataFrame], params: Optional[Dict[str, Any]] = None) -> float:
    if params is None:
        params = {
            "iterations": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "depth": trial.suggest_int("depth", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        }

    model = StackedModels()
    model.train(full_dataset=dataset_dict, **params)

    mae = model.evaluate(X_dict=dataset_dict, y_dict=dataset_dict)
    return mae


def hyperparameters_tuning():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    best_params = study.best_params
    print("Best hyperparameters:", best_params)
