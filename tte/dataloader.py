from dataclasses import dataclass, field, InitVar
from typing import Union, Any, Callable, Optional, Type, Protocol
from abc import abstractmethod
from pathlib import Path

import statsmodels.api as sm
import pandas as pd
import numpy as np
from lifelines.datasets import load_rossi
from rich import print

from tte import utils


class Dataloader(Protocol):
    """Basic representation of a dataloader"""

    @abstractmethod
    def load_data(self, name: str) -> Any:
        pass


class LinelifesDataloader(Dataloader):
    _dataset_factories = {"rossi": load_rossi}

    def load_data(self, name) -> Any:
        if name not in self._dataset_factories:
            raise KeyError(f"Dataset {name} not found.")
        df = self._dataset_factories[name]()
        return df


class StatsmodelsSurvivalDataloader(Dataloader):
    _dataset_packages = {"cancer": "survival", "mastectomy": "HSAUR", "heart": "survival"}

    def load_data(self, name) -> Any:
        if name not in self._dataset_packages:
            raise KeyError(f"Dataset {name} not found.")
        df = sm.datasets.get_rdataset(name, self._dataset_packages[name]).data
        return df

@dataclass
class AllDataloader(Dataloader):
    dataloaders: list[Dataloader]

    def load_data(self, name) -> Any:
        for dataloader in self.dataloaders:
            try:
                df = dataloader.load_data(name)
                return df
            except KeyError:
                continue
        else:
            raise KeyError(f"Dataset {name} not found.")


all_dataloader = AllDataloader(dataloaders=[LinelifesDataloader(), StatsmodelsSurvivalDataloader()])


dataset_map = {
    'cancer': ['status', 'time', ["sex", "age"], 2]
}

def get_sample(dataset_name="cancer"):
    path = Path(f"{utils.get_project_root()}/data/{dataset_name}.pickle")
    if path.is_file():
        return pd.read_pickle(path)

    dataloader = all_dataloader
    df = dataloader.load_data(dataset_name)
    df.to_pickle(path)
    return df


def main():
    dataset_name = 'cancer'
    dataloader = all_dataloader
    df = dataloader.load_data(dataset_name)
    print(df)

    df = get_sample(dataset_name=dataset_name)
    print(df)


if __name__ == "__main__":
    main()
