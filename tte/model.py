from dataclasses import dataclass, field, InitVar
from typing import Union, Any, Callable, Optional, Type, Protocol
from abc import abstractmethod

import statsmodels.api as sm
import pandas as pd
import numpy as np

from lifelines import CoxPHFitter
from rich import print
import matplotlib.pyplot as plt


class Model(Protocol):
    """Basic representation of a model"""

    @abstractmethod
    def fit(self, data, time_col: str, event_col: str) -> None:
        pass

    @abstractmethod
    def print_summary(self) -> None:
        pass

    @abstractmethod
    def plot(self, covariate: str, values: list) -> None:
        pass


class LinelifesModel(Model):
    _model_factories = {"cox": CoxPHFitter}

    def __init__(self, model: str):
        if model not in self._model_factories:
            raise KeyError(f"Model {model} not found.")
        self.model = self._model_factories[model]()

    def fit(self, data, time_col: str, event_col: str) -> None:
        self.model.fit(data, duration_col=time_col, event_col=event_col)

    def print_summary(self) -> None:
        self.model.print_summary()

    def plot(self, covariate: str, values: list) -> None:
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        self.model.plot(ax=ax1)
        self.model.plot_partial_effects_on_outcome(
            covariates=covariate, values=values, cmap="coolwarm", ax=ax2
        )
        plt.tight_layout()
        plt.show()


def load_test_dataset() -> Any:
    from lifelines.datasets import load_rossi

    return load_rossi()


def lifelines_main():
    df = load_test_dataset()
    model = LinelifesModel(model="cox")
    model.fit(
        data=df,
        time_col="week",
        event_col="arrest",
    )
    model.print_summary()
    model.plot(
        covariate="prio",
        values=list(range(0, 10, 2)),
    )


def main():
    pass


if __name__ == "__main__":
    lifelines_main()
