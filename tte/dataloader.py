from dataclasses import dataclass, field, InitVar
from typing import Union, Any, Callable, Optional, Type, Protocol
from abc import abstractmethod
from pathlib import Path

import statsmodels.api as sm
import pandas as pd
import numpy as np
from lifelines.datasets import load_rossi, load_regression_dataset
from rich import print

from tte import utils


def load_hk_covid():
    filename = 'hk_covid.csv'
    filepath = f"{utils.get_project_root()}/data/{filename}"
    df = pd.read_csv(filepath)
    df = df.drop(['dummy1', 'dummy2', 'dummy3'], axis=1)
    df = df.drop('Case no.', axis=1)
    df = df.loc[df['status'].isin(['Discharged', 'Deceased'])]
    df['residentship'] = df['residentship']=='HK resident'
    df['status'] = df['status']=='Deceased'
    df[['day', 'month', 'year']] = df['Report date'].str.split('/', expand=True)
    df['start'] = pd.to_datetime(df['Date of onset'], format='%d/%m/%Y', errors='coerce')
    df['end'] = pd.to_datetime(df['Report date'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna()
    df['year'] = df['year'].astype(int) - 2020
    df['season'] = df['month'].astype(int)%12 // 3 + 1
    df['time'] = (df['end'] - df['start']).dt.days + 1
    df = df.drop(['day', 'month', 'Report date', 'Date of onset', 'start', 'end', 'residentship'], axis=1)
    observed_df = df[df['status']]
    censored_df = df[~df['status']].sample(len(observed_df)//4)
    df = pd.concat([observed_df, censored_df])
    df['status'] = df['status'].astype(int)
    df['gender'] = np.where(df['gender']=='M', 1, 0)
    df['age'] = df['age'].astype(int)
    return df


class Dataloader(Protocol):
    """Basic representation of a dataloader"""

    @abstractmethod
    def load_data(self, name: str) -> Any:
        pass


class LinelifesDataloader(Dataloader):
    _dataset_factories = {"rossi": load_rossi, 'synthetic': load_regression_dataset}

    def load_data(self, name) -> Any:
        if name not in self._dataset_factories:
            raise KeyError(f"Dataset {name} not found.")
        df = self._dataset_factories[name]()
        return df


class StatsmodelsSurvivalDataloader(Dataloader):
    _dataset_packages = {"cancer": "survival"}

    def load_data(self, name) -> Any:
        if name not in self._dataset_packages:
            raise KeyError(f"Dataset {name} not found.")
        df = sm.datasets.get_rdataset(name, self._dataset_packages[name]).data
        return df

class CustomDataloader(Dataloader):
    _dataset_factories = {"covid": load_hk_covid}

    def load_data(self, name) -> Any:
        if name not in self._dataset_factories:
            raise KeyError(f"Dataset {name} not found.")
        df = self._dataset_factories[name]()
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


all_dataloader = AllDataloader(dataloaders=[CustomDataloader(), LinelifesDataloader(), StatsmodelsSurvivalDataloader()])


dataset_map = {
    'cancer': ['status', 'time', ["sex", "age"], 2],
    'transplant': ['death', 'time', ['age', 'black_male', 'white_male', 'black_female'], 1],
    'rossi': ['arrest', 'week', ['fin', 'age', 'race', 'wexp', 'mar', 'paro', 'prio'], 1],
    'synthetic': ['E', 'T', ['var1', 'var2', 'var3'], 1],
    'covid': ['status', 'time', ['gender', 'age', 'year', 'season'], 1]
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
    dataset_name = 'covid'
    dataloader = all_dataloader
    df = dataloader.load_data(dataset_name)
    print(df)


if __name__ == "__main__":
    main()
