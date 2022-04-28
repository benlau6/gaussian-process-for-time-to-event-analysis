import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from rich import print
from lifelines import KaplanMeierFitter

from tte.dataloader import StatsmodelsSurvivalDataloader
from tte.models.pymc_model import LogLogisticModel, LogLogisticGPModel
from tte import utils

# import jax
# from jax import numpy as jnp


def get_sample():
    dataloader = StatsmodelsSurvivalDataloader()
    df = dataloader.load_data("cancer")
    # df = dataloader.load_data("mastectomy")
    return df


def plot_sample(df, event_col, duration_col, event_val=1):
    fig, ax = plt.subplots(figsize=(8, 6))

    df = df.iloc[:20]

    ax.hlines(
        y=df.index,
        xmin=0,
        xmax=df[duration_col],
        color="k",
    )

    ax.scatter(
        x=df.loc[df[event_col] == event_val, duration_col],
        y=df.loc[df[event_col] == event_val].index,
        color="r",
        marker="x",
        label="event occurred",
    )

    ax.set_xlim(left=0)
    ax.set_xlabel("Months since observation")
    ax.set_yticks(df.index)
    ax.set_ylabel("Case")
    ax.legend(loc="best")


def main():
    df = get_sample()
    df = df.sample(50)
    event_col = "status"
    duration_col = "time"
    covariate_cols = ["sex", "age"]
    event_val = 2
    plot_sample(df, event_col="status", duration_col="time", event_val=event_val)

    X_df = df.loc[df[event_col] == event_val, covariate_cols]
    X_cens_df = df.loc[df[event_col] != event_val, covariate_cols]

    X = df.loc[df[event_col] == event_val, covariate_cols].to_numpy()
    y = df.loc[df[event_col] == event_val, duration_col].to_numpy()

    X_cens = df.loc[df[event_col] != event_val, covariate_cols].to_numpy()
    y_cens = df.loc[df[event_col] != event_val, duration_col].to_numpy()

    print(df)

    coords = {
        "total_cases": df.index,
        "event_cases": X_df.index,
        "censored_cases": X_cens_df.index,
    }
    idata_kwargs = {
        "dims": {
            "eta": ["total_cases"],
            "eta_rotated_": ["total_cases"],
        }
    }
    model = LogLogisticGPModel(X, y, X_cens, y_cens, coords=coords)
    g = pm.model_to_graphviz(model.model)
    g.render(filename=f"{utils.get_project_root()}/results/model", format="png")
    model.fit(n_samples=1000, idata_kwargs=idata_kwargs)

    az.plot_trace(model.idata, compact=True)
    plt.tight_layout()
    plt.savefig(f"{utils.get_project_root()}/results/mcmc_sampling.png")

    kmf = KaplanMeierFitter()
    kmf.fit(df[duration_col], df[event_col])

    fig, ax = plt.subplots(figsize=(8, 6))
    kmf.plot(ax=ax)
    # plt.savefig("km_survival_plot.png")
    model.plot_sf(X=[1, 1], ax=ax)
    plt.savefig(f"{utils.get_project_root()}/results/multiple_models_survival_plot.png")
    plt.show()


if __name__ == "__main__":
    main()
