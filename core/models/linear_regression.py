import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from core.embedding_models import DummyEmbeddingModel
from core.models.base_model import BaseMatchingModel


class LinearRegressionModel(BaseMatchingModel):
    def __init__(self, metric=metrics.mean_absolute_error, embedding_model=None):
        self.model = LinearRegression()
        self.metric = metric
        self.embedding_model = embedding_model

    def train(
        self, dataset: pd.DataFrame, test_size: float = 0.2, seed: int = 42, **kwargs
    ) -> Any:
        embeddings = dataset.emb.to_numpy()
        embeddings = np.array(list(map(np.array, embeddings)))
        target = dataset.target.to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, target, test_size=test_size, random_state=seed
        )

        self.model.fit(X_train, y_train, **kwargs)

        print("Model trained")

        y_pred = self.model.predict(X_test)
        test_score = self.metric(y_test, y_pred)
        print(f"Test score is {test_score}")

    def predict(self, vacancy: str | np.ndarray) -> float:
        if self.embedding_model is None:
            self.embedding_model = DummyEmbeddingModel()

        if isinstance(vacancy, str):
            emb = self.embedding_model.generate(vacancy)
        elif isinstance(vacancy, np.ndarray):
            emb = vacancy
        else:
            raise Exception("Wrong type of vacancy")

        return self.model.predict([emb])

    def save_model(self, path: Path | str) -> None:
        with open(str(path), "wb") as f:
            pickle.dump(self.model, f)
        print(f"Model saved at {path}")

    def load_model(self, path: Path | str) -> None:
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        print("Model successfully loaded")
