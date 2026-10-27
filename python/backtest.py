"""
Walk-forward backtest of the FIR halving-cycle model.

At each test date t (weekly) inside Cycle 3 and Cycle 4:
  * only prices up to t are used (earlier cycles complete, current cycle cut at t)
  * the filter is refit (closed-form ridge, same objective as the gradient-descent version)
  * forecast = filter path change over the next HORIZON days
  * compared with the realised log return over [t, t+HORIZON]

Baselines:  random walk (0),  trailing-1y drift,  cycle analog (mean realised move of
earlier cycles over the same phase window).

Run: python backtest.py [path/to/btc.csv]
"""
import sys
import numpy as np
import pandas as pd

URL = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
HAL = [pd.Timestamp(d) for d in ["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20", "2028-04-17"]]
N, HORIZON, STEP = 200, 60, 7
L2, SMOOTH = 1e-3, 1e-1
MIN_OBS_DAYS = 120          # need this much of the current cycle before testing


def load(path=URL):
    d = pd.read_csv(path, usecols=["time", "PriceUSD"], parse_dates=["time"]).dropna().set_index("time")["PriceUSD"]
    d = d.loc[HAL[0]:]
    return d.reindex(pd.date_range(d.index[0], d.index[-1])).ffill()


def cycle_signal(prices, k, cutoff):
    s, e = HAL[k], HAL[k + 1]
    seg = prices.loc[s:min(e, cutoff)]
    ph = np.asarray((seg.index - s) / (e - s))
    lv = np.log(seg.values / seg.values[0])
    g = np.linspace(0, 1, N)
    out = np.interp(g, ph, lv)
    out[g > ph[-1]] = np.nan
    return out


def conv_matrix(x, taps):
    half = taps // 2
    xp = np.pad(x, half, mode="edge")
    return np.stack([xp[n:n + taps][::-1] for n in range(N)])


def fit(pairs, taps):
    A, y = [], []
    for x, t in pairs:
        M = conv_matrix(x, taps)
        ok = ~np.isnan(t)
        A.append(M[ok]); y.append(t[ok])
    A, y = np.vstack(A), np.concatenate(y)
    m = len(y)
    Phi = np.hstack([A, np.ones((m, 1))])
    half = taps // 2
    r = np.abs(np.arange(taps) - half) / max(half, 1)
    W = L2 * (1 + 200 * r ** 4)
    D = np.diff(np.eye(taps), 2, axis=0)
    P = np.zeros((taps + 1, taps + 1))
    P[:taps, :taps] = np.diag(W) + SMOOTH * D.T @ D
    theta = np.linalg.solve(Phi.T @ Phi / m + P, Phi.T @ y / m)
    return theta[:taps], theta[taps]


def interp_at(path, phase):
    return np.interp(phase, np.linspace(0, 1, N), path)


def forecast_logret(prices, k, t, taps):
    """Model forecast of log return over [t, t+HORIZON] for a date t in cycle k."""
    sigs = [cycle_signal(prices, j, t) for j in range(k + 1)]
    pairs = [(sigs[j], sigs[j + 1]) for j in range(k)]
    h, b = fit(pairs, taps)
    path = conv_matrix(sigs[k - 1], taps) @ h + b
    s, e = HAL[k], HAL[k + 1]
    p0 = (t - s) / (e - s)
    p1 = (t + pd.Timedelta(days=HORIZON) - s) / (e - s)
    return interp_at(path, p1) - interp_at(path, p0)


def analog_logret(prices, k, t):
    """Mean realised move of earlier completed cycles over the same phase window."""
    s, e = HAL[k], HAL[k + 1]
    f0, f1 = (t - s) / (e - s), (t + pd.Timedelta(days=HORIZON) - s) / (e - s)
    rs = []
    for j in range(k):
        sj, ej = HAL[j], HAL[j + 1]
        a, c = (sj + (ej - sj) * f0).normalize(), (sj + (ej - sj) * f1).normalize()
        rs.append(np.log(prices.loc[c] / prices.loc[a]))
    return float(np.mean(rs))


def run(prices, taps_list):
    last = prices.index[-1]
    rows = []
    for k in (2, 3):  # Cycle 3 and Cycle 4 (0-indexed)
        t = HAL[k] + pd.Timedelta(days=MIN_OBS_DAYS)
        while t + pd.Timedelta(days=HORIZON) <= min(last, HAL[k + 1]):
            real = np.log(prices.loc[t + pd.Timedelta(days=HORIZON)] / prices.loc[t])
            drift = np.log(prices.loc[t] / prices.loc[t - pd.Timedelta(days=365)]) / 365 * HORIZON
            row = dict(date=t, cycle=k + 1, realised=real, rw=0.0, drift=drift, analog=analog_logret(prices, k, t))
            for taps in taps_list:
                row[f"fir{taps}"] = forecast_logret(prices, k, t, taps)
            rows.append(row)
            t += pd.Timedelta(days=STEP)
    return pd.DataFrame(rows)


def score(df, col):
    e = df[col] - df["realised"]
    rmse = np.sqrt((e ** 2).mean())
    return dict(model=col, n=len(df), rmse=rmse, mae=e.abs().mean(),
                corr=np.corrcoef(df[col], df["realised"])[0, 1] if df[col].std() > 0 else np.nan,
                hit=(np.sign(df[col]) == np.sign(df["realised"])).mean() if col != "rw" else np.nan)


def table(df, taps_list):
    cols = ["rw", "drift", "analog"] + [f"fir{t}" for t in taps_list]
    res = pd.DataFrame([score(df, c) for c in cols]).set_index("model")
    res["skill_vs_rw"] = 1 - res["rmse"] ** 2 / res.loc["rw", "rmse"] ** 2
    return res


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else URL
    prices = load(path)
    taps_list = [11, 21, 41, 61, 81]
    df = run(prices, taps_list)
    df.to_csv("backtest_results.csv", index=False)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}", "display.width", 140)
    print(f"Data through {prices.index[-1]:%Y-%m-%d} | horizon {HORIZON}d | step {STEP}d | {len(df)} test dates")
    print("\nALL TEST DATES (Cycles 3+4)"); print(table(df, taps_list).to_string())
    for c in (3, 4):
        sub = df[df.cycle == c]
        print(f"\nCYCLE {c} ONLY ({len(sub)} dates, {sub.date.min():%Y-%m-%d} to {sub.date.max():%Y-%m-%d})")
        print(table(sub, taps_list).to_string())
