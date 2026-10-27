"""
Rolling (cycle-free) model: no resets, no cycle template.
  features at date t : trailing log-price path  x_j = log P(t - 7j) - log P(t),  j = 1..W   (lags in weeks)
                       optionally + halving-phase features (cos/sin of position in the 4-year cycle)
  target             : log P(t+H) - log P(t)
  fit                : ridge with smoothness across lags, trained on every earlier date whose target is already
                       known at t (purged walk-forward, nothing from the future).
"""
import sys
import numpy as np, pandas as pd
import backtest as b, model

H = 90


def phase_feats(t):
    k = max(i for i in range(4) if b.HAL[i] <= t)
    ph = (t - b.HAL[k]) / (b.HAL[k + 1] - b.HAL[k])
    a = 2 * np.pi * ph
    return np.array([np.cos(a), np.sin(a), np.cos(2 * a), np.sin(2 * a)])


def build(lp, dates, W, use_phase):
    X, idx = [], []
    for t in dates:
        lag = [t - pd.Timedelta(days=7 * j) for j in range(1, W + 1)]
        if lag[-1] < lp.index[0]: continue
        f = lp.loc[lag].values - lp.loc[t]
        if use_phase: f = np.concatenate([f, phase_feats(t)])
        X.append(f); idx.append(t)
    return np.array(X), pd.DatetimeIndex(idx)


def fit_ridge(X, y, W, alpha, smooth, n_extra):
    n, k = X.shape; xm, ym = X.mean(0), y.mean(); Xc, yc = X - xm, y - ym
    P = alpha * np.eye(k)
    D = np.diff(np.eye(W), 2, axis=0); P[:W, :W] += smooth * D.T @ D
    w = np.linalg.solve(Xc.T @ Xc / n + P, Xc.T @ yc / n); return w, ym - xm @ w


def run(path, W=52, alpha=1.0, smooth=1.0, use_phase=False, clip=model.CLIP):
    p = b.load(path); pc = model.clean(p, clip) if clip else p; lp = np.log(pc); last = p.index[-1]
    all_dates = pd.date_range(pd.Timestamp("2020-09-08") - pd.Timedelta(days=7 * 400), last - pd.Timedelta(days=H), freq="7D")
    all_dates = all_dates[all_dates >= lp.index[0] + pd.Timedelta(days=7 * W)]
    X, idx = build(lp, all_dates, W, use_phase)
    y = np.array([np.log(p.loc[t + pd.Timedelta(days=H)] / p.loc[t]) for t in idx])
    yc_ = np.array([lp.loc[t + pd.Timedelta(days=H)] - lp.loc[t] for t in idx])     # target on the cleaned path for training
    rows = []
    for i, t in enumerate(idx):
        if t < pd.Timestamp("2020-09-08"): continue
        tr = np.where(idx + pd.Timedelta(days=H) <= t)[0]                          # purge: target must be known at t
        w, b0 = fit_ridge(X[tr], yc_[tr], W, alpha, smooth, 4 if use_phase else 0)
        rows.append(dict(t=t, pred=float(X[i] @ w + b0), real=y[i]))
    return pd.DataFrame(rows)


def score(df):
    e = df.pred - df.real; rw = np.sqrt((df.real ** 2).mean())
    return dict(n=len(df), skill=1 - (e ** 2).mean() / rw ** 2, corr=np.corrcoef(df.pred, df.real)[0, 1], hit=(np.sign(df.pred) == np.sign(df.real)).mean())


if __name__ == "__main__":
    import json
    path = sys.argv[1] if len(sys.argv) > 1 else b.URL
    tr = json.load(open("trend.json"))["taps"]["41"]
    cyc = pd.DataFrame(dict(t=pd.to_datetime(tr["d"], unit="ms"), cycle=tr["p"], real=tr["r"])).set_index("t")
    pd.set_option("display.float_format", lambda v: f"{v:.3f}", "display.width", 120)
    out = {}
    for W in (26, 52, 104):
        for alpha in (0.3, 3.0):
            for ph in (False, True):
                d = run(path, W=W, alpha=alpha, use_phase=ph).set_index("t")
                j = d.join(cyc[["cycle"]], how="inner")             # dates the cycle model also covers
                out[f"W{W} a{alpha} {'phase' if ph else 'trail'}"] = dict(**score(d), skill_on_cycle_dates=score(j)["skill"] if len(j) else np.nan)
    res = pd.DataFrame(out).T
    print("ROLLING MODEL, walk-forward 90d, all weekly dates 2020-09..2026 (no halving gap)"); print(res.to_string())
    best = run(path, W=52, alpha=3.0, use_phase=True).set_index("t")
    j = best.join(cyc[["cycle"]], how="inner")
    print(f"\nSame dates only (n={len(j)}), where the cycle model also has a forecast:")
    print(pd.DataFrame({"cycle model": score(j.assign(pred=j.cycle)), "rolling W52 a3 +phase": score(j), "average of the two": score(j.assign(pred=(j.pred + j.cycle) / 2))}).T.to_string())
    print("\nDates the cycle model cannot test (halving gap):", int((~best.index.isin(cyc.index)).sum()), "-> rolling model scores", {k: round(v, 3) for k, v in score(best[~best.index.isin(cyc.index)]).items()})
