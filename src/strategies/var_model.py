"""VAR(p) model with OLS fitting, BIC lag selection and model diagnostics.

x_t = c + A_1 x_{t-1} + ... + A_p x_{t-p} + e_t

Diagnostics:
    stability   - all eigenvalues of the companion matrix inside the unit circle
    portmanteau - multivariate (Hosking / adjusted Ljung-Box) test for residual autocorrelation
    ljung_box   - univariate Ljung-Box test per variable

Usage:
    model = VARmodel(num_vars=5, max_lag=10)
    model.load_training_data(returns)   # array of shape [time, features]
    model.fit()
    model.check()                       # uses the BIC-optimal lag unless lag= is given
"""

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import chi2


class trainingdata:  # feed dataset as [timeaxis, features]
    """Lagged design data for a VAR.

    data[i, m-1, :]  = x_{t-m}   (lag m = 1..max_lag)
    validation[i, :] = x_t       with t = i + max_lag
    Every lag order uses the same effective sample, so BIC values are comparable.
    """

    def __init__(self, data: np.ndarray, max_lag: int = 10):
        data = np.asarray(data, dtype=float)
        windows = sliding_window_view(data, max_lag + 1, axis=0)  # (T-L, N, L+1), windows[i,:,j] = x[i+j]
        self.data = np.transpose(windows[:, :, :-1][:, :, ::-1], (0, 2, 1))  # (T-L, L, N), newest lag first
        self.validation = data[max_lag:]

    def __getitem__(self, idx):
        return self.data[idx]

    def design(self, lag: int) -> np.ndarray:
        """Regressor matrix [1, x_{t-1}, ..., x_{t-lag}] of shape (T_eff, 1 + lag*N)."""
        X = self.data[:, :lag, :].reshape(self.data.shape[0], -1)
        return np.hstack([np.ones((X.shape[0], 1)), X])


class VARmodel:
    """VAR(p) fitted by OLS for every p = 1..max_lag."""

    def __init__(self, num_vars, max_lag=10):
        self.num_vars = num_vars
        self.max_lag = max_lag
        self.intercepts = [None] * max_lag  # (N,)
        self.coefs = [None] * max_lag       # (p, N, N), coefs[p-1][m-1] = A_m
        self.residuals = [None] * max_lag   # (T_eff, N)
        self.log_Sigma = [0.0] * max_lag

    def load_training_data(self, data: np.ndarray):
        self.training_data = trainingdata(data, max_lag=self.max_lag)

    def fit(self):
        Y = self.training_data.validation
        N = self.num_vars
        for lag in range(1, self.max_lag + 1):
            X = self.training_data.design(lag)
            B, *_ = np.linalg.lstsq(X, Y, rcond=None)  # (1 + lag*N, N)
            resid = Y - X @ B
            self.intercepts[lag - 1] = B[0]
            # B[1 + (m-1)N : 1 + mN] maps x_{t-m} -> x_t, so A_m is its transpose
            self.coefs[lag - 1] = B[1:].reshape(lag, N, N).transpose(0, 2, 1)
            self.residuals[lag - 1] = resid
            sigma = resid.T @ resid / resid.shape[0]
            _, logdet = np.linalg.slogdet(sigma)
            self.log_Sigma[lag - 1] = logdet

    @property
    def predictors(self):
        return list(zip(self.intercepts, self.coefs))

    def optimal_lag(self, verbose: bool = True):
        """Lag order minimising BIC."""
        logS = np.array(self.log_Sigma)
        T = self.training_data.validation.shape[0]
        BIC = logS + np.log(T) * (np.arange(1, self.max_lag + 1) * self.num_vars**2) / T
        best = int(np.argmin(BIC)) + 1
        if verbose:
            print("optimal lag:", best)
        return best

    def predict(self, history: np.ndarray, lag: int) -> np.ndarray:
        """One-step forecast from the last `lag` rows of history (shape [time, N])."""
        recent = np.asarray(history, dtype=float)[-lag:][::-1]  # x_{t-1}, ..., x_{t-p}
        return self.intercepts[lag - 1] + np.einsum("mij,mj->i", self.coefs[lag - 1], recent)

    # ------------------------------------------------------------------ diagnostics

    def companion_matrix(self, lag: int) -> np.ndarray:
        """(N*p x N*p) companion matrix [[A_1 ... A_p], [I 0]]."""
        N = self.num_vars
        F = np.zeros((N * lag, N * lag))
        F[:N, :] = np.hstack(self.coefs[lag - 1])
        F[N:, :-N] = np.eye(N * (lag - 1))
        return F

    def stability(self, lag: int):
        """Stable iff every eigenvalue of the companion matrix lies strictly inside the unit circle."""
        eig = np.linalg.eigvals(self.companion_matrix(lag))
        max_mod = float(np.max(np.abs(eig)))
        return {"stable": max_mod < 1.0, "max_modulus": max_mod, "eigenvalues": eig}

    def portmanteau(self, lag: int, h: int = None, adjusted: bool = True):
        """Multivariate portmanteau test (Hosking / adjusted Ljung-Box) on the residuals.

        H0: no residual autocorrelation up to lag h.  Q ~ chi2(N^2 (h - p)).
        """
        h = h if h is not None else lag + 10
        if h <= lag:
            raise ValueError("h must be larger than the VAR lag order")
        u = self.residuals[lag - 1]
        u = u - u.mean(axis=0)
        T, N = u.shape
        C0_inv = np.linalg.inv(u.T @ u / T)
        Q = 0.0
        for j in range(1, h + 1):
            Cj = u[j:].T @ u[:-j] / T
            term = np.trace(Cj.T @ C0_inv @ Cj @ C0_inv)
            Q += term / (T - j) if adjusted else term / T
        Q *= T**2
        df = N**2 * (h - lag)
        return {"statistic": Q, "df": df, "pvalue": float(chi2.sf(Q, df)), "h": h}

    def ljung_box(self, lag: int, h: int = None):
        """Univariate Ljung-Box test per variable. Q ~ chi2(h - p) (approximate df)."""
        h = h if h is not None else lag + 10
        u = self.residuals[lag - 1]
        u = u - u.mean(axis=0)
        T = u.shape[0]
        denom = np.sum(u**2, axis=0)
        k = np.arange(1, h + 1)
        r = np.array([np.sum(u[j:] * u[:-j], axis=0) / denom for j in k])  # (h, N)
        Q = T * (T + 2) * np.sum(r**2 / (T - k)[:, None], axis=0)
        df = max(h - lag, 1)
        return {"statistic": Q, "df": df, "pvalue": chi2.sf(Q, df), "h": h}

    def check(self, lag: int = None, h: int = None, alpha: float = 0.05):
        """Run stability + residual autocorrelation checks and print a summary."""
        lag = lag if lag is not None else self.optimal_lag()
        stab = self.stability(lag)
        port = self.portmanteau(lag, h)
        lb = self.ljung_box(lag, h)

        print(f"VAR({lag}) diagnostics")
        print(f"  Stability: max |eigenvalue| = {stab['max_modulus']:.4f} -> "
              f"{'STABLE' if stab['stable'] else 'NOT STABLE'}")
        print(f"  Portmanteau (h={port['h']}): Q = {port['statistic']:.2f}, df = {port['df']}, "
              f"p = {port['pvalue']:.4f} -> "
              f"{'no residual autocorrelation' if port['pvalue'] > alpha else 'residual autocorrelation remains'}")
        for i, (q, p) in enumerate(zip(lb["statistic"], lb["pvalue"])):
            print(f"  Ljung-Box var {i} (h={lb['h']}): Q = {q:.2f}, p = {p:.4f}"
                  f"{'' if p > alpha else '  <-- autocorrelated'}")

        passed = stab["stable"] and port["pvalue"] > alpha
        return {"lag": lag, "passed": passed, "stability": stab, "portmanteau": port, "ljung_box": lb}


# ---------------------------------------------------------------------- strategy

def forecast(
    prices: pl.DataFrame,
    horizon: int = 5,
    max_lag: int = 4,
    refit_every: int = 250,
    min_train: int = 520,
) -> pl.DataFrame:
    """Forecast of the next `horizon`-day (weekly) log return from a VAR on weekly log returns.

    Fitting: non-overlapping `horizon`-day log returns, anchored at the first row, on an expanding
    window. The lag is picked by BIC (up to `max_lag` weeks). The model is refitted every
    `refit_every` days, and each fit only uses prices up to and including its fit date.

    Forecasting: every day T, the fitted coefficients are applied to the rolling `horizon`-day log
    returns ending at T, T-horizon, ..., so the forecast updates daily. Rows stay null until
    `min_train` days are available.

    Args:
        prices: ``AsOfDate`` plus one price column per asset.
        horizon: Trading days per "week".
        max_lag: Largest VAR lag (in weeks) considered by BIC.
        refit_every: Days between refits.
        min_train: Days of history before the first fit.

    Returns:
        ``AsOfDate`` plus the predicted next-week log return per asset.
    """
    assets = [c for c in prices.columns if c != "AsOfDate"]
    logp = np.log(prices.select(assets).fill_null(strategy="forward").to_numpy().astype(float))
    n, N = logp.shape

    rolling = np.full_like(logp, np.nan)  # rolling[t] = log return over (t-horizon, t]
    rolling[horizon:] = logp[horizon:] - logp[:-horizon]

    out = np.full((n, N), np.nan)
    for start in range(min_train, n, refit_every):
        ends = np.arange(horizon, start + 1, horizon)  # week ends <= start, fixed grid from row 0
        weekly = logp[ends] - logp[ends - horizon]
        weekly = weekly[~np.isnan(weekly).any(axis=1)]

        model = VARmodel(num_vars=N, max_lag=max_lag)
        model.load_training_data(weekly)
        model.fit()
        lag = model.optimal_lag(verbose=False)

        stop = min(start + refit_every, n)
        days = np.arange(start, stop)
        pred = np.tile(model.intercepts[lag - 1], (len(days), 1))
        for m, A in enumerate(model.coefs[lag - 1], start=1):
            pred += rolling[days - horizon * (m - 1)] @ A.T
        out[start:stop] = pred

    return prices.select("AsOfDate").with_columns(
        pl.Series(asset, out[:, i]).fill_nan(None) for i, asset in enumerate(assets)
    )
