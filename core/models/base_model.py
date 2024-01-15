from abc import ABC, abstractmethod

import numpy as np


class BaseMatchingModel(ABC):
    @abstractmethod
    def predict(self, vacancy: str | np.ndarray) -> float:
        pass
