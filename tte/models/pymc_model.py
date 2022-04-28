from typing import Optional

import pymc as pm
import arviz as az
import aesara.tensor as at
import numpy as np
import matplotlib.pyplot as plt
import aesara


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class YLogLinkMixIn:
    def preprocess_y(self, y):
        y = np.log(y)
        self.logged = True
        return super().preprocess_y(y)


class YNormalizeMixIn:
    def preprocess_y(self, y):
        y = (y - y.mean()) / y.std()
        self.normalized = True
        self.y_mean = y.mean()
        self.y_std = y.std()
        return super().preprocess_y(y)


class XAddConstantMixIn:
    def preprocess_X(self, X):
        X = np.insert(X, 0, values=1, axis=1)
        return super().preprocess_X(X)


class PyMCModel:
    def __init__(self, X, y, X_cens=None, y_cens=None, coords=None):
        self.X = self.preprocess_X(X)
        self.y = self.preprocess_y(y)
        self.X_cens = self.preprocess_X(X_cens)
        self.y_cens = self.preprocess_y(y_cens)
        self.model = pm.Model(coords=coords)
        self.idata: Optional[az.InferenceData] = None
        self.define_model()

    def preprocess_X(self, X):
        return X

    def preprocess_y(self, y):
        return y

    def define_model(self):
        ...

    def sf(self):
        ...

    def sf_post(self, ts, X):
        ...

    def fit(self, n_samples=1000, n_tune=1000, n_chains=4, idata_kwargs=None):
        if idata_kwargs is None:
            idata_kwargs = {}
        with self.model:
            self.idata = pm.sample(
                n_samples,
                tune=n_tune,
                chains=n_chains,
                target_accept=0.99,
                return_inferencedata=True,
                idata_kwargs=idata_kwargs,
            )

    def sample_posterior_predictive(self):
        with self.model:
            self.idata.extend(pm.sample_posterior_predictive(self.idata))

    def get_var_post_mean(self, var_name):
        return self.idata.posterior[var_name].mean(dim=("chain", "draw")).data

    def plot_sf(self, sfs=None, ts=None, X=None, n_samples=10000):
        ...


class LogLogisticModel(XAddConstantMixIn, YLogLinkMixIn, PyMCModel):
    def sf(self, y, mu, sigma):
        # note it is logsitic sf
        return 1.0 - sigmoid(np.subtract.outer(y, mu) / sigma)

    def define_model(self):
        with self.model:
            beta = pm.Normal("beta", 0.0, sigma=5, shape=self.X.shape[1])
            sigma = pm.HalfNormal("sigma", 5.0)

            y_obs = pm.Logistic("y_obs", at.dot(beta, self.X.T), sigma, observed=self.y)
            if (self.y_cens is not None) and (self.X_cens is not None):
                y_cens = pm.Potential(
                    "y_cens", self.sf(self.y_cens, at.dot(beta, self.X_cens.T), sigma)
                )

    def plot_sf(self, X, t_max=1000, n_samples=10000, ax=None):
        ts = np.linspace(0, t_max, n_samples)
        X = np.array(X)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        beta = self.get_var_post_mean("beta")
        sigma = self.get_var_post_mean("sigma")

        sfs = self.sf(
            y=np.log(ts),
            mu=beta @ X.T,
            sigma=sigma,
        )

        ax.plot(ts, sfs, label="Log-Logistic AFT")
        ax.legend()

        betas = self.idata.posterior["beta"].mean(dim=("chain")).data
        eta = betas @ np.array([1, 1])
        sfss = self.sf(y=np.log(ts), mu=eta, sigma=sigma)
        az.plot_hdi(ts, sfss.T, smooth=False, ax=ax)


class LogLogisticGPModel(XAddConstantMixIn, YLogLinkMixIn, PyMCModel):
    def sf(self, y, mu, sigma):
        # note it is logsitic sf
        return 1.0 - sigmoid(np.subtract.outer(y, mu) / sigma)

    def define_model(self):
        with self.model:
            # beta = pm.Normal("beta", 0.0, sigma=5, shape=self.X.shape[1])
            sigma = pm.HalfNormal("sigma", 5.0)

            l = pm.HalfNormal("l", 5)
            cov_func = pm.gp.cov.Matern52(input_dim=self.X.shape[1], ls=l)
            gp = pm.gp.Latent(cov_func=cov_func)
            eta = gp.prior("eta", X=np.concatenate([self.X, self.X_cens]))
            # eta0 = gp.prior("eta0", X=self.X)
            # eta1 = gp.prior("eta1", X=self.X_cens)

            # eta0, eta1 = at.split(eta, splits_size=self.X.shape[0], n_splits=1)
            eta0 = eta[: self.X.shape[0]]
            eta1 = eta[self.X.shape[0] :]

            sf = pm.Deterministic(
                "sf",
                1 - pm.invlogit((self.y_cens - eta1) / sigma),
                dims="censored_cases",
            )

            y_obs = pm.Logistic(
                "y_obs", eta0, sigma, observed=self.y, dims="event_cases"
            )
            if (self.y_cens is not None) and (self.X_cens is not None):
                # y_cens = pm.Potential("y_cens", self.sf(self.y_cens, beta, sigma))
                y_cens = pm.Potential("y_cens", sf)

    def plot_sf(self, X, t_max=1000, n_samples=10000, ax=None):
        ts = np.linspace(0, t_max, n_samples)
        X = np.array(X)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        eta = self.get_var_post_mean("eta")
        sigma = self.get_var_post_mean("sigma")

        sfs = self.sf(
            y=np.log(ts),
            mu=eta,
            sigma=sigma,
        )

        ax.plot(ts, sfs, label="Log-Logistic GP AFT")
        ax.legend()

        eta = self.idata.posterior["eta"].mean(dim=("chain")).data
        # sfss = self.sf(y=np.log(ts), mu=eta, sigma=sigma)
        # az.plot_hdi(ts, sfss.T, smooth=False, ax=ax)
