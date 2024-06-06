import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import dill as pickle
import numpy as np
import optuna
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from typing_extensions import Self

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
        embedding = np.atleast_2d(embedding)
        return self.model.predict(embedding)[0]

    def save_model(self, path: Path | str) -> None:
        with open(str(path), "wb") as f:
            pickle.dump(self.model, f)
        print(f"Model saved at {path}")

    def load_model(self, path: Path | str) -> Self:
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        print("Model successfully loaded")
        return self


class StackedModels(BaseMatchingModel):
    def __init__(
        self,
        base_model_class: type[CatboostRegressionModel] = CatboostRegressionModel,
        clustering_model: Optional[ClusteringModel] = None,
        **model_kwargs,
    ):
        self.base_model_class = base_model_class
        self.base_model_params = {**model_kwargs}

        self.clustering_model = clustering_model
        self.models: Dict[int, BaseMatchingModel] = {}

    @staticmethod
    def retrieve_cluster_data(full_dataset: pd.DataFrame, cluster_label: int) -> pd.DataFrame:
        if "cluster_label" not in full_dataset.columns:
            raise KeyError("cluster_label not found in the DataFrame")

        df_cluster = full_dataset[full_dataset["cluster_label"] == cluster_label]
        return df_cluster

    @staticmethod
    def split_dict_dataset(
        dataset_dict: Dict[int, pd.DataFrame],
        test_size: float = 0.2,
        seed: int = 42,
    ) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
        return {
            cluster_label: train_test_split(
                dataset,
                test_size=test_size,
                random_state=seed,
            )
            for cluster_label, dataset in dataset_dict.items()
        }

    @staticmethod
    def _check_dataset_format(dataset: pd.DataFrame) -> None:
        necessary_columns = ("emb", "target")
        for column in necessary_columns:
            if column not in dataset.columns:
                raise KeyError(f"{column} not in the DataFrame")

    def train(
        self,
        dataset_dict: Dict[int, pd.DataFrame],
        test_size: float = 0.2,
        **train_kwargs,
    ) -> Self:
        if test_size > 0.0:
            split_dataset_dict = self.split_dict_dataset(dataset_dict, test_size=test_size)
        else:
            split_dataset_dict = {k: (v, None) for k, v in dataset_dict.items()}

        for cluster_label, (dataset_train, _) in split_dataset_dict.items():
            model = self.base_model_class(**self.base_model_params)
            model.train(dataset=dataset_train, test_size=0.0, **train_kwargs)
            self.models[cluster_label] = model

        print("Model trained")

        if test_size > 0.0:
            test_score = self.evaluate({k: v for k, (_, v) in split_dataset_dict.items()})
            print(f"Test score is {test_score}")

        return self

    def evaluate(self, dataset_dict: Dict[int, pd.DataFrame]) -> float:
        cluster_metrics = {}
        for cluster_label, model in self.models.items():
            dataset = dataset_dict[cluster_label]
            self._check_dataset_format(dataset)

            embeddings = np.array(list(map(np.array, dataset.emb.to_numpy())))
            target = dataset.target.to_numpy()
            X, y_true = embeddings, target

            y_pred = model.predict(X)
            cluster_metrics[cluster_label] = mean_absolute_error(y_true, y_pred)

        print(json.dumps(cluster_metrics, indent=4))
        return np.mean(list(cluster_metrics.values()))

    def predict(self, embedding: np.ndarray, cluster_label: Optional[int] = None) -> float:
        if not (self.clustering_model or cluster_label is not None):
            raise ValueError("Provide either clusterization model or cluster label")

        if cluster_label is None:
            cluster_label = self.clustering_model.predict(embedding)

        return self.models[cluster_label].predict(embedding)

    def save_model(self, path: Path | str) -> None:
        with open(str(path), "wb") as f:
            pickle.dump(self.models, f)
        print(f"Model saved at {path}")

    def load_model(self, path: Path | str) -> Self:
        with open(path, "rb") as f:
            self.models = pickle.load(f)
        print("Model successfully loaded")
        return self


def objective(
    trial,
    dataset_dict: Dict[int, pd.DataFrame],
    params: Optional[Dict[str, Any]] = None,
) -> float:
    if params is None:
        params = {
            "iterations": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "depth": trial.suggest_int("depth", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        }

    model = StackedModels(**params)

    split_dataset_dict = model.split_dict_dataset(dataset_dict)
    train_dataset_dict = {k: v for k, (v, _) in split_dataset_dict.items()}
    test_dataset_dict = {k: v for k, (_, v) in split_dataset_dict.items()}

    model = model.train(dataset_dict=train_dataset_dict)
    mae = model.evaluate(dataset_dict=test_dataset_dict)
    return mae


def hyperparameters_tuning(dataset_dict: Dict[int, pd.DataFrame]) -> Dict[str, Any]:
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda x: objective(x, dataset_dict=dataset_dict), n_trials=20)

    best_params = study.best_params
    print("Best hyperparameters:", best_params)
    return best_params
