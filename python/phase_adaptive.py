"""
Phase-adaptive FIR: the filter taps (and bias) vary smoothly along the cycle phase axis.
At each phase n the taps are a locally-weighted ridge fit (Gaussian kernel of width `bw` in
phase) over all training pairs, shrunk toward the global filter with strength `kappa`
(partial pooling): kappa -> inf recovers the global filter, kappa -> 0 is fully local.
"""
import numpy as np, pandas as pd
import backtest as b

N = b.N


def build(pairs, taps):
    A, y, idx = [], [], []
    for x, t in pairs:
        M = b.conv_matrix(x, taps); ok = ~np.isnan(t)
        A.append(np.hstack([M, np.ones((N, 1))])[ok]); y.append(t[ok]); idx.append(np.arange(N)[ok])
    return np.vstack(A), np.concatenate(y), np.concatenate(idx)


def penalty(taps):
    half = taps // 2
    r = np.abs(np.arange(taps) - half) / max(half, 1); W = b.L2 * (1 + 200 * r ** 4)
    D = np.diff(np.eye(taps), 2, axis=0)
    P = np.zeros((taps + 1, taps + 1)); P[:taps, :taps] = np.diag(W) + b.SMOOTH * D.T @ D
    return P


def global_theta(A, y, P):
    m = len(y); return np.linalg.solve(A.T @ A / m + P, A.T @ y / m)


def local_theta(A, y, idx, P, th_g, n, bw, kappa):
    m = len(y); w = np.exp(-0.5 * ((idx - n) / (bw * N)) ** 2)
    Aw = A * w[:, None]
    K = A.shape[1]
    lhs = Aw.T @ A / m + P + kappa * np.eye(K)
    rhs = Aw.T @ y / m + kappa * th_g
    return np.linalg.solve(lhs, rhs)


def forecast(prev_sig, pairs, taps, phi0, phi1, bw, kappa):
    A, y, idx = build(pairs, taps); P = penalty(taps)
    th_g = global_theta(A, y, P)
    M = np.hstack([b.conv_matrix(prev_sig, taps), np.ones((N, 1))])
    cache = {}
    def val(n):
        if n not in cache:
            th = th_g if kappa is None else local_theta(A, y, idx, P, th_g, n, bw, kappa)
            cache[n] = M[n] @ th
        return cache[n]
    def at(phi):
        g = phi * (N - 1); n0 = int(np.clip(np.floor(g), 0, N - 2)); f = g - n0
        return val(n0) * (1 - f) + val(n0 + 1) * f
    return at(phi1) - at(phi0)


def run(horizons=(60, 90), taps=41, grid=((0.08, 0.1), (0.08, 1), (0.15, 1), (0.15, 10), (0.3, 1), (0.3, 10)), step=14, path=None):
    p = b.load(path); last = p.index[-1]; rows = []
    for H in horizons:
        for k in (2, 3):
            s, e = b.HAL[k], b.HAL[k + 1]; t = b.HAL[k] + pd.Timedelta(days=b.MIN_OBS_DAYS)
            while t + pd.Timedelta(days=H) <= min(last, e):
                sigs = [b.cycle_signal(p, j, t) for j in range(k + 1)]; pairs = [(sigs[j], sigs[j + 1]) for j in range(k)]
                ph = lambda d: (t + pd.Timedelta(days=d) - s) / (e - s)
                r = dict(H=H, cycle=k + 1, real=np.log(p.loc[t + pd.Timedelta(days=H)] / p.loc[t]))
                r["global"] = forecast(sigs[k - 1], pairs, taps, ph(0), ph(H), None, None)
                for bw, kap in grid:
                    r[f"bw{bw}_k{kap}"] = forecast(sigs[k - 1], pairs, taps, ph(0), ph(H), bw, kap)
                rows.append(r); t += pd.Timedelta(days=step)
    return pd.DataFrame(rows)


def score(g, c):
    e = g[c] - g.real; rw = np.sqrt((g.real ** 2).mean())
    return dict(skill=1 - (e ** 2).mean() / rw ** 2, corr=np.corrcoef(g[c], g.real)[0, 1], hit=(np.sign(g[c]) == np.sign(g.real)).mean(), n=len(g))


if __name__ == "__main__":
    import sys
    d = run(path=sys.argv[1] if len(sys.argv) > 1 else b.URL)
    d.to_csv("phase_adaptive_results.csv", index=False)
    cols = [c for c in d.columns if c not in ("H", "cycle", "real")]
    pd.set_option("display.float_format", lambda v: f"{v:.3f}", "display.width", 120)
    for H in (60, 90):
        for lab, sub in [("all", d[d.H == H]), ("cycle 4", d[(d.H == H) & (d.cycle == 4)]), ("cycle 3", d[(d.H == H) & (d.cycle == 3)])]:
            print(f"\nH={H}d {lab} (n={len(sub)})"); print(pd.DataFrame({c: score(sub, c) for c in cols}).T.drop(columns="n").to_string())
