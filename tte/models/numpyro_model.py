import functools
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from tinygp import kernels, GaussianProcess
from jax import numpy as jnp
from jax import scipy as jsp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import arviz as az
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
from sksurv.metrics import brier_score
from sksurv.util import Surv

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

    if y is not None:
        y = y_func(y)
    return X_func(X), y


class SurvivalModel:
    _model_name = ''
    _model_shortname = ''

    def model(self, X, y=None, X_cens=None, y_cens=None):
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

    def model(self, X, y=None, X_cens=None, y_cens=None, **kwargs):
        # add constant to X as a baseline, take log on y for modelling log logistic
        preprocessing_ = functools.partial(
            preprocessing, X_func=X_add_constant, y_func=jnp.log
        )

        X, y = preprocessing_(X, y)
        # apply the same transformation on censored data
        if (X_cens is not None) and (y_cens is not None):
            X_cens, y_cens = preprocessing_(X_cens, y_cens)

        loc = numpyro.sample("loc", dist.Normal(0.0, 5.0), sample_shape=(1, X.shape[1]))

        # Add some noise to observation
        scale = numpyro.sample("scale", dist.HalfNormal(5.0))

        # Finally, our observation model is Logistic
        y_obs = numpyro.sample("obs", dist.Logistic(loc @ X.T, scale), obs=y)

        # censored
        if ((X_cens is not None) and (y_cens is not None)) and (y_cens is not None):
            constraint = 1 - dist.Logistic(loc @ X_cens.T, scale).cdf(y_cens)
            numpyro.factor("log_censored_sf", constraint)

    def sf(self, ts, X, loc, scale):
        # constant term
        X_ = X_add_constant(X)
        return jsp.stats.logistic.sf((jnp.log1p(ts) - (loc @ X_.T).T) / scale)

    def sfs(self, idata, ts, X, loc=None, scale=None):
        loc = idata.posterior["loc"].mean(dim=("chain", "draw")).data
        scale = idata.posterior["scale"].mean(dim=("chain", "draw")).data
        sfs = self.sf(ts, X, loc, scale)
        return sfs

def build_gp(rho, noise_gp, X):
    gp = GaussianProcess(
        kernels.Matern52(rho),
        X,
        diag=noise_gp,
    )
    return gp

class LogLogisticGPModel(SurvivalModel):
    _model_name = 'Log-Logistic GP'
    _model_shortname = 'llgp'

    def model(self, X, y=None, X_cens=None, y_cens=None, **kwargs):
        # take log on y for modelling log logistic
        preprocessing_ = functools.partial(
            preprocessing, X_func=None, y_func=jnp.log
        )

        X, y = preprocessing_(X, y)
        # apply the same transformation on censored data
        if (X_cens is not None) and (y_cens is not None):
            X_cens, y_cens = preprocessing_(X_cens, y_cens)

        if (X_cens is not None) and (y_cens is not None):
            X_ = jnp.vstack([X, X_cens])
        else:
            X_ = X

        # The parameters of the GP model
        rho = numpyro.sample("rho", dist.HalfNormal(10.0))
        noise_gp = numpyro.sample("noise_gp", dist.HalfNormal(1.0))

        if 'gp_cond' in kwargs:
            # prediction
            loc = numpyro.deterministic("loc", kwargs['gp_cond'].gp.mean)
        else:
            gp = build_gp(rho, noise_gp, X_)
            loc = numpyro.sample("loc", gp.numpyro_dist())


        loc1 = loc[: X.shape[0]]
        loc2 = loc[X.shape[0] :]

        # Finally, our observation model is Logistic
        scale = numpyro.sample("scale", dist.HalfNormal(5.0))

        y_obs = numpyro.sample("obs", dist.Logistic(loc=loc1, scale=scale), obs=y)

        # censored
        if (X_cens is not None) and (y_cens is not None):
            constraint = 1 - dist.Logistic(loc=loc2, scale=scale).cdf(y_cens)
            numpyro.factor("log_censored_sf", constraint)

    def sf(self, ts, loc, scale):
        return jsp.stats.logistic.sf((jnp.log1p(ts) - loc) / scale)

    def sfs(self, idata, ts, X=None, loc=None, scale=None):
        if loc is None:
            loc = idata.posterior["loc"].mean(dim=("chain", "draw")).data
        if scale is None:
            scale = idata.posterior["scale"].mean(dim=("chain", "draw")).data

        sfs = self.sf(ts, loc[:, jnp.newaxis], scale)
        return sfs


class LogLogisticChainedGPModel(SurvivalModel):
    _model_name = 'Log-Logistic Chained GP'
    _model_shortname = 'llcgp'

    def model(self, X, y=None, X_cens=None, y_cens=None, **kwargs):
        # take log on y for modelling log logistic
        preprocessing_ = functools.partial(
            preprocessing, X_func=None, y_func=jnp.log
        )

        X, y = preprocessing_(X, y)
        # apply the same transformation on censored data
        if (X_cens is not None) and (y_cens is not None):
            X_cens, y_cens = preprocessing_(X_cens, y_cens)

        if (X_cens is not None) and (y_cens is not None):
            X_ = jnp.vstack([X, X_cens])
        else:
            X_ = X

        # The parameters of the GP model
        rho1 = numpyro.sample("rho1", dist.HalfNormal(10.0))
        noise_gp1 = numpyro.sample("noise_gp1", dist.HalfNormal(1.0))

        if 'gp_cond1' in kwargs:
            # prediction
            loc = numpyro.deterministic("loc", kwargs['gp_cond1'].gp.mean)
        else:
            gp1 = build_gp(rho1, noise_gp1, X_)
            loc = numpyro.sample("loc", gp1.numpyro_dist())

        # second gp
        rho2 = numpyro.sample("rho2", dist.HalfNormal(10.0))
        noise_gp2 = numpyro.sample("noise_gp2", dist.HalfNormal(1.0))

        if 'gp_cond2' in kwargs:
            # prediction
            log_scale = numpyro.deterministic("log_scale", kwargs['gp_cond2'].gp.mean)
        else:
            gp2 = build_gp(rho2, noise_gp2, X_)
            log_scale = numpyro.sample("log_scale", gp2.numpyro_dist())

        # This parameter has shape (num_data,)
        loc1 = loc[: X.shape[0]]
        loc2 = loc[X.shape[0] :]

        # It will take exponential to ensure positivity
        log_scale1 = log_scale[: X.shape[0]]
        log_scale2 = log_scale[X.shape[0] :]

        # Finally, our observation model is Logistic
        y_obs = numpyro.sample(
            "obs", dist.Logistic(loc=loc1, scale=jnp.exp(log_scale1)), obs=y
        )

        # censored
        if (X_cens is not None) and (y_cens is not None):
            constraint = 1 - dist.Logistic(loc=loc2, scale=jnp.exp(log_scale2)).cdf(y_cens)
            numpyro.factor("log_censored_sf", constraint)

    def sf(self, ts, loc, scale):
        return jsp.stats.logistic.sf((jnp.log1p(ts) - loc) / jnp.exp(scale))

    def sfs(self, idata, ts, X=None, loc=None, scale=None):
        if loc is None:
            loc = idata.posterior["loc"].mean(dim=("chain", "draw")).data
        if scale is None:
            scale = idata.posterior["log_scale"].mean(dim=("chain", "draw")).data

        sfs = self.sf(ts, loc[:, jnp.newaxis], scale[:, jnp.newaxis])
        return sfs


def df2arr(df, event_col, duration_col, covariate_cols, event_val):
    # X need not to be np.nan in this stage
    X = df.loc[df[event_col] == event_val, covariate_cols].to_numpy()
    y = df.loc[df[event_col] == event_val, duration_col].to_numpy()

    X_cens = df.loc[df[event_col] != event_val, covariate_cols].to_numpy()
    y_cens = df.loc[df[event_col] != event_val, duration_col].to_numpy()
    return X, y, X_cens, y_cens

def combine_y(y, y_cens):
    return jnp.concatenate([y, y_cens])

def combine_X(X, X_cens):
    return jnp.vstack([X, X_cens])

def get_events(y, y_cens):
    return [*[1]*len(y), *[0]*len(y_cens)]


if __name__ == '__main__':
    numpyro.set_host_device_count(4)
    seed_num = 0
    rng_key = random.PRNGKey(seed_num)


    model_instance = LogLogisticModel()
    subsample = True
    subsample_size = 100
    num_warmup_mcmc = 5000
    num_samples_mcmc = 5000
    render_model = False
    only_render = False

    # cancer, rossi, synthetic
    dataset_name = 'synthetic'
    event_col, duration_col, covariate_cols, event_val = dataset_map[dataset_name]

    df = get_sample(dataset_name)
    if subsample:
        df = df.sample(subsample_size, random_state=rng_key)

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
        num_chains=4,
        progress_bar=True,
    )
    start = timer()
    mcmc.run(rng_key, X=X, y=y, X_cens=X_cens, y_cens=y_cens)
    end = timer()
    print(f"Sampling time: {end - start}s")
    #mcmc.print_summary()
    samples = mcmc.get_samples()

    # MCMC sampling trace

    idata = az.from_numpyro(mcmc)

    az.plot_trace(idata, figsize=(4, 8))
    plt.tight_layout()

    # Survival Plot

    ts = np.linspace(0, y.max(), 10000)

    fig, ax = plt.subplots(figsize=(8, 6))

    sfs = model_instance.sfs(idata, ts, X=X)

    for covariate, sf in zip(covariates, sfs):
        model_instance.plot_sf(covariate, sf)

    if len(sfs) < 10:
        ax.legend()
    ax.set_title(model_instance._model_name)
    plt.show()

    # testing
    def get_y_pred(predictions):
        mean_predictions = jnp.mean(predictions, axis=0)

        # difference between AFT and GPAFT
        if len(mean_predictions.shape) == 2:
            mean_predictions = mean_predictions[0]
        y_pred = jnp.expm1(mean_predictions)
        return y_pred

    def combine_y_result(y_true, y_pred, event_values):
        df = pd.DataFrame([y_true, y_pred]).T
        df.columns = ['true', 'pred']
        df['event'] = event_values
        return df

    X, y, X_cens, y_cens = df2arr(df, event_col, duration_col, covariate_cols, event_val)
    event_values = [*[1]*len(y), *[0]*len(y_cens)]
    X_ = jnp.vstack([X, X_cens])
    y_true = np.concatenate([y, y_cens])

    # build gp
    gp_cond = None
    gp_cond1 = None
    gp_cond2 = None
    loc_cond = None
    scale_cond = None
    if isinstance(model_instance, LogLogisticGPModel):
        post_loc = jnp.mean(samples['loc'], axis=0)
        post_rho = jnp.mean(samples['rho'], axis=0)
        post_noise_gp = jnp.mean(samples['noise_gp'], axis=0)
        gp = build_gp(post_rho, post_noise_gp, covariates)
        gp_cond = gp.condition(post_loc, X_)
        loc_cond = gp_cond.gp.mean
    elif isinstance(model_instance, LogLogisticChainedGPModel):
        post_loc = jnp.mean(samples['loc'], axis=0)
        post_rho1 = jnp.mean(samples['rho1'], axis=0)
        post_noise_gp1 = jnp.mean(samples['noise_gp2'], axis=0)

        post_log_scale = jnp.mean(samples['log_scale'], axis=0)
        post_rho2 = jnp.mean(samples['rho2'], axis=0)
        post_noise_gp2 = jnp.mean(samples['noise_gp2'], axis=0)

        gp1 = build_gp(post_rho1, post_noise_gp1, covariates)
        gp_cond1 = gp1.condition(post_loc, X_)
        loc_cond = gp_cond1.gp.mean

        gp2 = build_gp(post_rho2, post_noise_gp2, covariates)
        gp_cond2 = gp2.condition(post_log_scale, X_)
        scale_cond = gp_cond2.gp.mean


    predictive = Predictive(model_instance.model, samples)
    predictions = predictive(rng_key, X=X_, gp_cond=gp_cond, gp_cond1=gp_cond1, gp_cond2=gp_cond2)["obs"]
    y_pred = get_y_pred(predictions)
    df = combine_y_result(y_true, y_pred, event_values)
    # print(df)

    c_index = concordance_index(df['true'], df['pred'], df['event'])
    survival_test = Surv.from_dataframe('event', 'true', df)

    ts = np.linspace(y_true.min()+1, y_true.max()-1, 5)
    pred_sf = model_instance.sfs(idata, ts, X=X_, loc=loc_cond, scale=scale_cond)

    b_score = brier_score(survival_test, survival_test, pred_sf, np.linspace(y_true.min()+1, y_true.max()-1, 5))
    print(f"{c_index:.3f}")
    print(b_score[1].round(3))
