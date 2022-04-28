import functools
import warnings

# to suppress mac jax experimental warning
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from tinygp import kernels, GaussianProcess
from jax import numpy as jnp
from jax import scipy as jsp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az
import matplotlib.pyplot as plt

from tte import utils
from tte.dataloader import get_sample, dataset_map


def X_add_constant(X):
    return jnp.insert(X, 0, values=1, axis=1)


def preprocessing(X, y, X_func, y_func):
    id_func = lambda x: x
    if X_func is None:
        X_func = id_func
    if y_func is None:
        y_func = id_func

    return X_func(X), y_func(y)


class SurvivalModel:
    _model_name = ''
    _model_shortname = ''

    def model(self, X, y, X_cens=None, y_cens=None):
        raise NotImplementedError

    def sf(self):
        raise NotImplementedError

    def sfs(self):
        raise NotImplementedError

    def plot_sf(self, covariates, sf):
        label = f"{self._model_name} with {covariates}"
        ax.plot(ts, sf, label=label)

    def render_model(self, X, y, X_cens, y_cens, render_distributions, render_params):
        # print model architecture
        g = numpyro.render_model(
            self.model,
            model_kwargs={
                "X": X,
                "y": y,
                "X_cens": X_cens,
                "y_cens": y_cens,
            },
            render_distributions=render_distributions,
            render_params=render_params,
        )

        g.render(
            filename=f"{utils.get_project_root()}/results/{self._model_shortname}_model{'_dists' if render_distributions else ''}",
            format="png",
            engine="dot",
        )

class LogLogisticModel(SurvivalModel):
    _model_name = 'Log-Logistic AFT'
    _model_shortname = 'll'

    def model(self, X, y, X_cens=None, y_cens=None):
        # add constant to X as a baseline, take log on y for modelling log logistic
        X, y = preprocessing(X, y, X_func=X_add_constant, y_func=jnp.log)

        # apply the same transformation on censored data
        if (X_cens is not None) and (y_cens is not None):
            X_cens, y_cens = preprocessing(
                X_cens, y_cens, X_func=X_add_constant, y_func=jnp.log
            )

        loc = numpyro.sample("loc", dist.Normal(0.0, 5.0), sample_shape=(1, X.shape[1]))

        # Add some noise to observation
        scale = numpyro.sample("scale", dist.HalfNormal(5.0))

        # Finally, our observation model is Logistic
        y_obs = numpyro.sample("obs", dist.Logistic(loc @ X.T, scale), obs=y)

        # censored
        if (y_cens is not None) and (X_cens is not None):
            constraint = 1 - dist.Logistic(loc @ X_cens.T, scale).cdf(y_cens)
            numpyro.factor("log_censored_sf", constraint)

    def sf(self, ts, X, loc, scale):
        # constant term
        X_ = X_add_constant(X)
        return jsp.stats.logistic.sf((jnp.log1p(ts) - (loc @ X_.T).T) / scale)

    def sfs(self, ts, X, idata):
        loc = idata.posterior["loc"].mean(dim=("chain", "draw")).data
        scale = idata.posterior["scale"].mean(dim=("chain", "draw")).data
        sfs = self.sf(ts, X, loc, scale)
        return sfs


class LogLogisticGPModel(SurvivalModel):
    _model_name = 'Log-Logistic GP'
    _model_shortname = 'llgp'

    def model(self, X, y, X_cens=None, y_cens=None):
        # add constant to X as a baseline, take log on y for modelling log logistic

        X, y = preprocessing(X, y, X_func=X_add_constant, y_func=jnp.log)

        # apply the same transformation on censored data
        if (X_cens is not None) and (y_cens is not None):
            X_cens, y_cens = preprocessing(
                X_cens, y_cens, X_func=X_add_constant, y_func=jnp.log
            )

        # The parameters of the GP model
        noise_gp = numpyro.sample("noise_gp", dist.HalfNormal(1.0))
        rho = numpyro.sample("rho", dist.HalfNormal(10.0))
        kernel = kernels.Matern52(rho)
        gp = GaussianProcess(
            kernel,
            jnp.vstack([X, X_cens]),
            diag=noise_gp,
        )

        # This parameter has shape (num_data,)
        loc = numpyro.sample("loc", gp.numpyro_dist())

        loc1 = loc[: X.shape[0]]
        loc2 = loc[X.shape[0] :]

        # Finally, our observation model is Logistic
        scale = numpyro.sample("scale", dist.HalfNormal(5.0))

        y_obs = numpyro.sample("obs", dist.Logistic(loc=loc1, scale=scale), obs=y)

        # censored
        if (y_cens is not None) and (X_cens is not None):
            constraint = 1 - dist.Logistic(loc=loc2, scale=scale).cdf(y_cens)
            numpyro.factor("log_censored_sf", constraint)

    def sf(self, ts, loc, scale):
        return jsp.stats.logistic.sf((jnp.log1p(ts) - loc) / scale)

    def sfs(self, ts, idata, X=None):
        locs = idata.posterior["loc"].mean(dim=("chain", "draw")).data
        scale = idata.posterior["scale"].mean(dim=("chain", "draw")).data

        sfs = self.sf(ts, locs[:, jnp.newaxis], scale)
        return sfs


class LogLogisticChainedGPModel(SurvivalModel):
    _model_name = 'Log-Logistic Chained GP'
    _model_shortname = 'llcgp'

    def model(self, X, y, X_cens=None, y_cens=None):
        # add constant to X as a baseline, take log on y for modelling log logistic
        preprocessing_ = functools.partial(
            preprocessing, X_func=X_add_constant, y_func=jnp.log
        )

        X, y = preprocessing_(X, y)

        # apply the same transformation on censored data
        if (X_cens is not None) and (y_cens is not None):
            X_cens, y_cens = preprocessing_(X_cens, y_cens)

        # The parameters of the GP model
        noise_gp1 = numpyro.sample("noise_gp1", dist.HalfNormal(1.0))
        rho1 = numpyro.sample("rho1", dist.HalfNormal(10.0))
        kernel1 = kernels.Matern52(rho1)
        gp1 = GaussianProcess(
            kernel1,
            jnp.vstack([X, X_cens]),
            diag=noise_gp1,
        )

        # second gp
        noise_gp2 = numpyro.sample("noise_gp2", dist.HalfNormal(1.0))
        rho2 = numpyro.sample("rho2", dist.HalfNormal(10.0))
        kernel2 = kernels.Matern52(rho2)
        gp2 = GaussianProcess(
            kernel2,
            jnp.vstack([X, X_cens]),
            diag=noise_gp2,
        )

        # This parameter has shape (num_data,)
        loc = numpyro.sample("loc", gp1.numpyro_dist())
        loc1 = loc[: X.shape[0]]
        loc2 = loc[X.shape[0] :]

        # It will take exponential to ensure positivity
        log_scale = numpyro.sample("log_scale", gp2.numpyro_dist())
        log_scale1 = log_scale[: X.shape[0]]
        log_scale2 = log_scale[X.shape[0] :]

        # Finally, our observation model is Logistic
        y_obs = numpyro.sample(
            "obs", dist.Logistic(loc=loc1, scale=jnp.exp(log_scale1)), obs=y
        )

        # censored
        if (y_cens is not None) and (X_cens is not None):
            constraint = 1 - dist.Logistic(loc=loc2, scale=jnp.exp(log_scale2)).cdf(y_cens)
            numpyro.factor("log_censored_sf", constraint)

    def sf(self, ts, loc, scale):
        return jsp.stats.logistic.sf((jnp.log1p(ts) - loc) / jnp.exp(scale))

    def sfs(self, ts, idata, X=None):
        locs = idata.posterior["loc"].mean(dim=("chain", "draw")).data
        log_scales = idata.posterior["log_scale"].mean(dim=("chain", "draw")).data

        sfs = self.sf(ts, locs[:, jnp.newaxis], log_scales[:, jnp.newaxis])
        return sfs


class PiecewisePHModel(SurvivalModel):
    _model_name = 'Piecewise Proportional Hazard Model'
    _model_shortname = 'pph'

    def model(self, X, y, X_cens=None, y_cens=None):
        # add constant to X as a baseline, take log on y for modelling log logistic
        X, y = preprocessing(X, y, X_func=lambda x: x, y_func=lambda x: x)

        # apply the same transformation on censored data
        if (X_cens is not None) and (y_cens is not None):
            X_cens, y_cens = preprocessing(
                X_cens, y_cens, X_func=X_add_constant, y_func=jnp.log
            )

        lambda0 = numpyro.sample("lambda0", dist.Gamma(0.01, 0.01), sample_shape=(1, X.shape[1]))

        beta = numpyro.sample("beta", dist.Normal(0, 1000))

        # Finally, our observation model is Logistic
        y_obs = numpyro.sample("obs", dist.Poisson(loc @ X.T, scale), obs=y)

    def sf(self, ts, X, loc, scale):
        # constant term
        X_ = X_add_constant(X)
        return jsp.stats.logistic.sf((jnp.log1p(ts) - (loc @ X_.T).T) / scale)

    def sfs(self, ts, X, idata):
        loc = idata.posterior["loc"].mean(dim=("chain", "draw")).data
        scale = idata.posterior["scale"].mean(dim=("chain", "draw")).data
        sfs = self.sf(ts, X, loc, scale)
        return sfs


def df2arr(df, event_col, duration_col, covariate_cols, event_val):
    # X need not to be np.nan in this stage
    X = df.loc[df[event_col] == event_val, covariate_cols].to_numpy()
    y = df.loc[df[event_col] == event_val, duration_col].to_numpy()

    X_cens = df.loc[df[event_col] != event_val, covariate_cols].to_numpy()
    y_cens = df.loc[df[event_col] != event_val, duration_col].to_numpy()
    return X, y, X_cens, y_cens


if __name__ == '__main__':
    rng_key = random.PRNGKey(0)
    model_instance = LogLogisticModel()
    subsample = True
    subsample_size = 20
    num_warmup_mcmc = 500
    num_samples_mcmc = 500
    render_model = False
    only_render = False

    dataset_name = 'cancer'
    df = get_sample(dataset_name)
    if subsample:
        df = df.sample(subsample_size, random_state=rng_key)
    event_col, duration_col, covariate_cols, event_val = dataset_map[dataset_name]

    X, y, X_cens, y_cens = df2arr(df, event_col, duration_col, covariate_cols, event_val)

    # for later plotting survival curve
    covariates = np.vstack([X, X_cens])

    if render_model:
        model_instance.render_model(
            X, y, X_cens, y_cens,
            render_distributions=True, render_params=True)

    if only_render:
        exit()

    # Run the MCMC
    nuts_kernel = NUTS(
        model_instance.model,
    )

    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup_mcmc,
        num_samples=num_samples_mcmc,
        # num_chains=4,
        # progress_bar=False,
    )
    mcmc.run(rng_key, X=X, y=y, X_cens=X_cens, y_cens=y_cens)
    mcmc.print_summary()  # doctest: +SKIP

    # MCMC sampling trace

    idata = az.from_numpyro(mcmc)

    az.plot_trace(idata, figsize=(4, 8))
    plt.tight_layout()

    # Survival Plot

    ts = np.linspace(0, 1000, 10000)

    fig, ax = plt.subplots(figsize=(8, 6))

    sfs = model_instance.sfs(ts, X, idata)

    model_instance.plot_sf(covariates[0], sfs[0])
    model_instance.plot_sf(covariates[1], sfs[1])
    ax.legend()
    plt.show()
