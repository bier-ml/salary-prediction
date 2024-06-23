from catboost import CatBoostRegressor
from sklearn import metrics

from core.models.linear_regression import LinearRegressionModel


class CatboostRegressionModel(LinearRegressionModel):
    def __init__(self, metric=metrics.mean_absolute_error, embedding_model=None, **regressor_kwargs):
        super().__init__(metric, embedding_model)
        self.metric = metric
        self.model = CatBoostRegressor(**regressor_kwargs)
        self.embedding_model = embedding_model
