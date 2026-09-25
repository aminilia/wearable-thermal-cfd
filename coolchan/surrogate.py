"""Design of experiments, surrogate models and Monte Carlo.

Physics used to shape the surrogate
-----------------------------------
With constant properties and no buoyancy the energy equation is linear in T and
the flow does not depend on T. Hence, for a fixed geometry/flow and conductivity,

    T_max - T_in = Q * R(U, H, k)          (R = thermal resistance, K/W)

exactly, and the pressure drop depends on (U, H) only. The "physics-informed"
surrogate therefore regresses log R on (U, H, k) and multiplies by Q, instead of
asking a generic regressor to rediscover linearity in Q. Both it and two
conventional surrogates (quadratic response surface, GP on T_max) are scored by
leave-one-out cross-validation and on independent hold-out CFD runs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# design / uncertain variables and their DOE ranges
BOUNDS = {
    "U_in": (0.5, 3.0),       # m/s  - fan setting
    "H_mm": (1.5, 4.0),       # mm   - channel gap (block is 1 mm tall)
    "Q_W": (0.2, 1.0),        # W    - block power
    "k_solid": (120., 240.),  # W/m/K - cast Al ... pure Al
}
NAMES = list(BOUNDS)


def lhs(n: int, seed: int) -> np.ndarray:
    """Space-filling Latin hypercube in physical units, shape (n, 4)."""
    s = qmc.LatinHypercube(d=len(BOUNDS), seed=seed, optimization="random-cd")
    u = s.random(n)
    lo = np.array([b[0] for b in BOUNDS.values()])
    hi = np.array([b[1] for b in BOUNDS.values()])
    return qmc.scale(u, lo, hi)


def _unit(X, cols):
    lo = np.array([BOUNDS[c][0] for c in cols])
    hi = np.array([BOUNDS[c][1] for c in cols])
    return (np.asarray(X, float) - lo) / (hi - lo)


def _gp(dim, seed=0):
    k = (ConstantKernel(1.0, (1e-3, 1e3))
         * Matern(length_scale=np.ones(dim), length_scale_bounds=(0.05, 20.0), nu=2.5)
         + WhiteKernel(1e-6, (1e-10, 1e-2)))
    return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=8,
                                    random_state=seed)


# --------------------------------------------------------------------------- #
class Surrogate:
    name = "base"
    cols = NAMES

    def fit(self, df):  # df: pandas DataFrame with NAMES + T_max_C, T_in_C
        raise NotImplementedError

    def predict(self, df, return_std=False):
        raise NotImplementedError


class QuadraticRSM(Surrogate):
    """Full quadratic response surface in all four inputs -> T_max."""

    name = "Quadratic RSM (T_max)"

    def fit(self, df):
        self.m = make_pipeline(PolynomialFeatures(2), LinearRegression())
        self.m.fit(_unit(df[self.cols], self.cols), df["T_max_C"])
        return self

    def predict(self, df, return_std=False):
        y = self.m.predict(_unit(df[self.cols], self.cols))
        return (y, np.zeros_like(y)) if return_std else y


class GPTmax(Surrogate):
    """Gaussian process in all four inputs -> T_max."""

    name = "GP (T_max)"

    def fit(self, df):
        self.m = _gp(4).fit(_unit(df[self.cols], self.cols), df["T_max_C"])
        return self

    def predict(self, df, return_std=False):
        return self.m.predict(_unit(df[self.cols], self.cols), return_std=return_std)


class GPResistance(Surrogate):
    """Physics-informed: GP on log R(U, H, k); T_max = T_in + Q * R."""

    name = "GP on log R, × Q (physics-informed)"
    cols = ["U_in", "H_mm", "k_solid"]

    def fit(self, df):
        R = (df["T_max_C"] - df["T_in_C"]) / df["Q_W"]
        self.m = _gp(3).fit(_unit(df[self.cols], self.cols), np.log(R))
        return self

    def predict_R(self, X, return_std=False):
        """X: array (n,3) of U, H, k. Returns R (and std of log R)."""
        mu, sd = self.m.predict(_unit(X, self.cols), return_std=True)
        return (np.exp(mu), sd) if return_std else np.exp(mu)

    def predict(self, df, return_std=False, T_in=None):
        T_in = df["T_in_C"].to_numpy() if T_in is None else T_in
        R, sd = self.predict_R(df[self.cols].to_numpy(), return_std=True)
        y = T_in + df["Q_W"].to_numpy() * R
        # delta method: std of T = Q * R * std(log R)
        return (y, df["Q_W"].to_numpy() * R * sd) if return_std else y


class GPPressure:
    """GP on log dp(U, H) (dp is independent of Q and k)."""

    cols = ["U_in", "H_mm"]

    def fit(self, df):
        self.m = _gp(2).fit(_unit(df[self.cols], self.cols), np.log(df["dp_Pa"]))
        return self

    def predict(self, X):
        return np.exp(self.m.predict(_unit(X, self.cols)))


def loo(model_cls, df):
    """Leave-one-out predictions."""
    pred = np.empty(len(df))
    for i in range(len(df)):
        tr = df.drop(df.index[i])
        pred[i] = model_cls().fit(tr).predict(df.iloc[[i]])[0]
    return pred


def metrics(y, yhat):
    e = np.asarray(yhat) - np.asarray(y)
    ss = ((np.asarray(y) - np.mean(y)) ** 2).sum()
    return dict(rmse=float(np.sqrt(np.mean(e**2))), max_abs=float(np.abs(e).max()),
                r2=float(1 - (e**2).sum() / ss))


# --------------------------------------------------------------------------- #
@dataclass
class Uncertainty:
    """Input uncertainty for the Monte Carlo (independent)."""

    Q_nom: float = 0.5        # W
    Q_cv: float = 0.10        # 10 % (1-sigma) load variation between units/use cases
    k_lo: float = 150.0       # W/m/K  uniform: alloy / temper not controlled
    k_hi: float = 220.0
    T_in_mean: float = 25.0   # C  ambient air entering the vent
    T_in_sd: float = 1.5
    T_limit: float = 43.0     # skin-contact comfort limit
    include_model_error: bool = True


def exceedance_probability(model: GPResistance, U, H, unc: Uncertainty, n=20000,
                           seed=1, risk=0.01):
    """Monte Carlo on the surrogate at each (U, H) design point.

    Returns
    -------
    P       probability that T_max exceeds ``unc.T_limit`` at the nominal load
    T95     95th percentile of T_max
    Q_allow largest nominal load [W] whose exceedance probability is <= ``risk``

    Because T_max = T_in + Q_nom (1 + cv z) R, the event T_max > T_limit is
    Q_nom > (T_limit - T_in) / ((1 + cv z) R); Q_allow is therefore the
    ``risk``-quantile of that ratio - exact, no bisection needed. The GP's own
    predictive uncertainty on log R is sampled too when requested.
    """
    rng = np.random.default_rng(seed)
    U = np.atleast_1d(U)
    H = np.atleast_1d(H)
    k = rng.uniform(unc.k_lo, unc.k_hi, n)
    zq = rng.standard_normal(n)
    Tin = rng.normal(unc.T_in_mean, unc.T_in_sd, n)
    z = rng.standard_normal(n)
    P, T95, Qa = (np.empty(U.shape) for _ in range(3))
    for idx in np.ndindex(U.shape):
        X = np.column_stack([np.full(n, U[idx]), np.full(n, H[idx]), k])
        R, sd = model.predict_R(X, return_std=True)
        if unc.include_model_error:
            R = R * np.exp(sd * z)
        B = np.clip(1 + unc.Q_cv * zq, 1e-6, None) * R      # T = T_in + Q_nom * B
        T = Tin + unc.Q_nom * B
        P[idx] = np.mean(T > unc.T_limit)
        T95[idx] = np.quantile(T, 0.95)
        Qa[idx] = np.quantile((unc.T_limit - Tin) / B, risk)
    return P, T95, Qa
