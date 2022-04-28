from dataclasses import dataclass, field, InitVar
from typing import Union, Any, Callable, Optional, Type, Protocol
from abc import abstractmethod

import statsmodels.api as sm
import pandas as pd
import numpy as np
from rich import print
from tte.dataloader import LinelifesDataloader, Dataloader
from tte.model import LinelifesModel, Model

# df = sm.datasets.get_rdataset("cancer", "survival").data


@dataclass
class Experiment:
    dataloader: Dataloader
    model: Model


@dataclass
class ExperimentFactory:
    dataloader_class: Type[Dataloader]
    model_class: Type[Model]

    def __call__(self):
        return Experiment(self.dataloader_class(), self.model_class())


FACTORIES = {"rossi": ExperimentFactory(LinelifesDataloader, LinelifesModel)}


def read_factory(dataset_name) -> ExperimentFactory:
    """Constructs an dataloader factory based on the user's preference."""

    if dataset_name not in FACTORIES:
        raise KeyError(f"Unknown dataset: {dataset_name}")

    return FACTORIES[dataset_name]


def do_experiment(experiment: Experiment) -> None:
    print(experiment)


def main():
    # create the factory
    factory = read_factory("rossi")

    # use the factory to create the experiment
    experiment = factory()

    do_experiment(experiment)


if __name__ == "__main__":
    main()
