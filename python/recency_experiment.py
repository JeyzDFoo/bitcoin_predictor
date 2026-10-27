"""
Recency weighting: training pair j (earlier cycle -> next cycle) gets weight gamma**age, age 0 = the
pair ending in the current (partially observed) cycle; that current pair is additionally boosted by beta.
gamma = 1, beta = 1 reproduces the unweighted fit.
"""
import sys
import numpy as np, pandas as pd
import backtest as b
from shock_experiment import clean


def wfit(pairs, weights, taps):
    A, y, w = [], [], []
    for (x, t), wt in zip(pairs, weights):
        M = b.conv_matrix(x, taps); ok = ~np.isnan(t)
        A.append(np.hstack([M, np.ones((b.N, 1))])[ok]); y.append(t[ok]); w.append(np.full(ok.sum(), wt))
    A, y, w = np.vstack(A), np.concatenate(y), np.concatenate(w)
    m = len(y)
    half = taps // 2; r = np.abs(np.arange(taps) - half) / max(half, 1)
    P = np.zeros((taps + 1, taps + 1)); D = np.diff(np.eye(taps), 2, axis=0)
    P[:taps, :taps] = np.diag(b.L2 * (1 + 200 * r ** 4)) + b.SMOOTH * D.T @ D
    wn = w / w.mean()                                   # keep overall data-vs-penalty balance comparable
    th = np.linalg.solve((A * wn[:, None]).T @ A / m + P, (A * wn[:, None]).T @ y / m)
    return th[:taps], th[taps]


def run(path, H, grid, clip, taps=41):
    p = b.load(path); last = p.index[-1]; pc = clean(p, clip) if clip else p; rows = []
    for k in (2, 3):
        s, e = b.HAL[k], b.HAL[k + 1]; t = s + pd.Timedelta(days=b.MIN_OBS_DAYS)
        while t + pd.Timedelta(days=H) <= min(last, e):
            sigs = [b.cycle_signal(pc, j, t) for j in range(k + 1)]
            pairs = [(sigs[j], sigs[j + 1]) for j in range(k)]
            ph = lambda d: (t + pd.Timedelta(days=d) - s) / (e - s)
            r = dict(cycle=k + 1, real=np.log(p.loc[t + pd.Timedelta(days=H)] / p.loc[t]))
            for g, bt in grid:
                wts = [g ** (k - 1 - j) * (bt if j == k - 1 else 1.0) for j in range(k)]
                h, bb = wfit(pairs, wts, taps); path_ = b.conv_matrix(sigs[k - 1], taps) @ h + bb
                r[f"g{g}_b{bt}"] = b.interp_at(path_, ph(H)) - b.interp_at(path_, ph(0))
            rows.append(r); t += pd.Timedelta(days=b.STEP)
    return pd.DataFrame(rows)


def score(g, c):
    e = g[c] - g.real; rw = np.sqrt((g.real ** 2).mean())
    return dict(skill=1 - (e ** 2).mean() / rw ** 2, corr=np.corrcoef(g[c], g.real)[0, 1], hit=(np.sign(g[c]) == np.sign(g.real)).mean())


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else b.URL
    grid = [(1.0, 1.0), (0.7, 1.0), (0.5, 1.0), (0.3, 1.0), (1.0, 3.0), (0.5, 3.0)]
    pd.set_option("display.float_format", lambda v: f"{v:.3f}", "display.width", 120)
    for clip in (None, 2.5):
        for H in (60, 90):
            d = run(path, H, grid, clip); cols = [c for c in d.columns if c.startswith("g")]
            print(f"\n=== horizon {H}d, shock clip = {clip} ===  (g = per-cycle decay, b = boost on current cycle)")
            for lab, sub in [("all", d), ("cycle 3", d[d.cycle == 3]), ("cycle 4", d[d.cycle == 4])]:
                print(f"\n{lab} (n={len(sub)})"); print(pd.DataFrame({c: score(sub, c) for c in cols}).T.to_string())
