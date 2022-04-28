import pandas as pd

from tte import utils

df = pd.read_csv(f"{utils.get_project_root()}/data/hk_covid.csv")
print(df)