"""
Shock handling: causally clip daily log returns that are > k rolling-sigmas (e.g. COVID crash),
rebuild the log-price path from the clipped returns, feed THAT to the filter, and score the
forecast against the real (unclipped) realised returns.
"""
import sys
import numpy as np, pandas as pd
import backtest as b


def clean(p, k, win=90):
    r = np.log(p).diff().fillna(0.0)
    sig = r.rolling(win, min_periods=30).std().shift(1)        # uses data strictly before the day
    lim = (k * sig).fillna(np.inf)
    rc = r.clip(lower=-lim, upper=lim)
    return np.exp(np.log(p.iloc[0]) + rc.cumsum())


def run(path, ks=(None, 4, 3, 2.5), H=90):
    p = b.load(path); last = p.index[-1]; b.HORIZON = H
    series = {k: (p if k is None else clean(p, k)) for k in ks}
    nshock = {k: int(((np.log(p).diff().abs() > k * np.log(p).diff().rolling(90, min_periods=30).std().shift(1)).sum())) for k in ks if k}
    rows = []
    for c in (2, 3):
        s, e = b.HAL[c], b.HAL[c + 1]; t = s + pd.Timedelta(days=b.MIN_OBS_DAYS)
        while t + pd.Timedelta(days=H) <= min(last, e):
            real = np.log(p.loc[t + pd.Timedelta(days=H)] / p.loc[t])
            r = dict(cycle=c + 1, real=real)
            for k in ks: r[str(k)] = b.forecast_logret(series[k], c, t, 41)
            rows.append(r); t += pd.Timedelta(days=b.STEP)
    return pd.DataFrame(rows), nshock


def score(g, c):
    e = g[c] - g.real; rw = np.sqrt((g.real ** 2).mean())
    return dict(skill=1 - (e ** 2).mean() / rw ** 2, corr=np.corrcoef(g[c], g.real)[0, 1], hit=(np.sign(g[c]) == np.sign(g.real)).mean())


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else b.URL
    pd.set_option("display.float_format", lambda v: f"{v:.3f}", "display.width", 120)
    for H in (60, 90):
        d, ns = run(path, H=H)
        print(f"\n=== horizon {H}d === shock days flagged (whole history): {ns}")
        for lab, sub in [("all", d), ("cycle 3", d[d.cycle == 3]), ("cycle 4", d[d.cycle == 4])]:
            print(f"\n{lab} (n={len(sub)}); columns = clip threshold in sigmas (None = no clipping)")
            print(pd.DataFrame({c: score(sub, c) for c in ['None', '4', '3', '2.5']}).T.to_string())
