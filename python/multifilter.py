"""
Three-clock model: halving-cycle filter + financial-quarter filter + low-to-peak swing filter,
with diminishing returns handled explicitly.  Everything is walk-forward: at each test date t only
prices up to t are used, and only outcomes already known by t are used to fit anything.

Components (each gives a forecast of the log return over the next H days):
  H0     current site model (FIR on halving phase, recency 0.5, shock clip 2.5 sigma, 41 taps)
  Hamp   same filter fitted on SHAPE (each cycle divided by its run-up size); size of the
         current cycle = decay model on market cap at the halving, updated from the observed path
  Hvol   same, but each cycle scaled by its own volatility instead of its run-up
  Q      day-of-quarter clock: mean vol-scaled forward return of past dates at the same point in a
         calendar quarter, times today's volatility
  S      swing clock: mean vol-scaled forward return of past dates with the same swing state
         (rising since a confirmed low / falling since a confirmed high) and similar days since it
Combination: Q and S are fitted on what the halving filter missed (residuals), then added with
weights learned only from earlier out-of-sample predictions, shrunk toward zero and kept >= 0.

Run: python multifilter.py btc.csv [outdir]
"""
import sys, json
import numpy as np
import pandas as pd

HAL = pd.to_datetime(["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20", "2028-04-17"])
N, TAPS, L2, SMOOTH, GAMMA, CLIP = 200, 41, 1e-3, 1e-1, 0.5, 2.5
WARMUP = {1: 120, 2: 120, 3: 30}
DAY = pd.Timedelta(days=1)


# ---------------------------------------------------------------- data
def load(path):
    d = pd.read_csv(path, usecols=["time", "PriceUSD", "CapMrktCurUSD"], parse_dates=["time"]).set_index("time")
    d = d[d.PriceUSD.notna()].loc["2011-01-01":]
    idx = pd.date_range(d.index[0], d.index[-1])
    return d.reindex(idx).ffill()


def clean_series(p, k=CLIP):
    """Causal shock clipping, same as the site: |daily log return| capped at k x trailing 90-day sigma."""
    r = np.diff(np.log(p), prepend=np.log(p[0])); r[0] = 0
    out = np.empty_like(r)
    for i in range(len(r)):
        lo = max(1, i - 90)
        if i - lo >= 30:
            s = r[lo:i].std(ddof=1); out[i] = np.clip(r[i], -k * s, k * s)
        else:
            out[i] = r[i]
    return np.exp(np.log(p[0]) + np.cumsum(out))


# ---------------------------------------------------------------- halving filter (port of the site's engine)
def cycle_sig(CL, t0, k, cutoff):
    s, e = HAL[k], HAL[k + 1]
    i0 = (s - t0).days
    i1 = min((e - t0).days, len(CL) - 1, (cutoff - t0).days)
    if i1 <= i0:
        return np.full(N, np.nan)
    ii = np.arange(i0, i1 + 1)
    ph = (ii - i0) / (e - s).days
    lv = np.log(CL[ii] / CL[i0])
    g = np.arange(N) / (N - 1)
    out = np.interp(g, np.asarray(ph, float), lv)
    out[g > float(ph[-1])] = np.nan
    return out


_CONV = {}
def conv(x, taps):
    if taps not in _CONV:
        half = taps // 2
        _CONV[taps] = np.clip(np.arange(N)[:, None] + half - np.arange(taps)[None, :], 0, N - 1)
    return x[_CONV[taps]]


def train(pairs, taps=TAPS):
    A, y, w = [], [], []
    for x, t, wt in pairs:
        M = conv(x, taps); ok = ~np.isnan(t)
        A.append(M[ok]); y.append(t[ok]); w.append(np.full(ok.sum(), wt))
    A, y, w = np.vstack(A), np.concatenate(y), np.concatenate(w)
    m = len(y); w = w / w.mean() / m
    Phi = np.hstack([A, np.ones((m, 1))])
    half = taps // 2
    r = np.abs(np.arange(taps) - half) / max(half, 1)
    P = np.zeros((taps + 1, taps + 1))
    D = np.diff(np.eye(taps), 2, axis=0)
    P[:taps, :taps] = np.diag(L2 * (1 + 200 * r ** 4)) + SMOOTH * D.T @ D
    P[taps, taps] = 1e-9
    th = np.linalg.solve(Phi.T @ (Phi * w[:, None]) + P, Phi.T @ (w * y))
    return th[:taps], th[taps]


def apply(x, h, b):
    return conv(x, len(h)) @ h + b


def at_phase(path, ph):
    return np.interp(ph, np.arange(N) / (N - 1), path)


def phase(t, k):
    return (t - HAL[k]).days / (HAL[k + 1] - HAL[k]).days


def h_forecast(sigs, k, t, H, scales=None, cur_scale=None):
    """Forward H-day log-return forecast from the halving filter.  With scales, every cycle is divided by
    its scale (shape-only fit) and the forecast is multiplied back by the current cycle's scale."""
    c = [s.copy() for s in sigs]
    if scales is not None:
        for j in range(k):
            c[j] = c[j] / scales[j]
        c[k] = c[k] / cur_scale
    pairs = [(c[j], c[j + 1], GAMMA ** (k - 1 - j)) for j in range(k)]
    h, b = train(pairs)
    proj = apply(c[k - 1], h, b)
    d = at_phase(proj, phase(t + H * DAY, k)) - at_phase(proj, phase(t, k))
    return d * (cur_scale if scales is not None else 1.0), proj


# ---------------------------------------------------------------- causal swing state
def swing_state(p, rev=0.20):
    """For each day: +1 if rising since a confirmed low, -1 if falling since a confirmed high (0 before the
    first turn), and the index of that confirmed pivot.  Uses only prices up to that day."""
    n = len(p); mode = np.zeros(n, int); piv = np.full(n, -1)
    m, ext, last = 0, 0, -1
    for i in range(1, n):
        if m == 0:
            if p[i] >= p[ext] * (1 + rev): m, last, ext = 1, ext, i
            elif p[i] <= p[ext] * (1 - rev): m, last, ext = -1, ext, i
            elif p[i] < p[ext]: ext = i
        elif m == 1:
            if p[i] > p[ext]: ext = i
            elif p[i] <= p[ext] * (1 - rev): m, last, ext = -1, ext, i
        else:
            if p[i] < p[ext]: ext = i
            elif p[i] >= p[ext] * (1 + rev): m, last, ext = 1, ext, i
        mode[i], piv[i] = m, last
    return mode, piv


# ---------------------------------------------------------------- main experiment
def run(path, H):
    df = load(path)
    P = df.PriceUSD.values; t0 = df.index[0]; dates = df.index
    lp = np.log(P)
    r = np.diff(lp, prepend=lp[0])
    vol = pd.Series(r).rolling(90).std().values           # trailing daily sigma
    vol365 = pd.Series(r).rolling(365).std().values
    CL = clean_series(P)
    mode, piv = swing_state(P)
    last = dates[-1]
    i_of = lambda t: (t - t0).days
    fwd = lambda t: lp[i_of(t + H * DAY)] - lp[i_of(t)]
    mcap = {k: df.CapMrktCurUSD.iloc[i_of(HAL[k])] for k in range(4)}

    # completed-cycle run-up sizes and volatilities (cleaned series, as the filter sees it)
    full = {k: cycle_sig(CL, t0, k, HAL[k + 1]) for k in range(3)}
    AMP = {k: np.nanmax(full[k]) for k in range(3)}
    i_h = {k: i_of(HAL[k]) for k in range(4)}
    VOLC = {k: np.std(np.diff(np.log(CL[i_h[k]:i_h[k + 1]]))) for k in range(3)}

    rows = []
    t = HAL[1] + WARMUP[1] * DAY
    while t + H * DAY <= last:
        k = int(np.searchsorted(HAL, t, side="right")) - 1
        if t + H * DAY > HAL[k + 1] or (t - HAL[k]).days < WARMUP[k]:
            t += 7 * DAY; continue
        i = i_of(t)
        sigs = [cycle_sig(CL, t0, j, t) for j in range(k + 1)]
        f0, _ = h_forecast(sigs, k, t, H)

        # --- Hamp: size prior from completed cycles: log(size) linear in log(market cap at halving)
        xs = np.log([mcap[j] for j in range(k)]); ys = np.log([AMP[j] for j in range(k)])
        prior = np.exp(np.polyval(np.polyfit(xs, ys, 1), np.log(mcap[k]))) if k >= 2 else AMP[k - 1]
        A = prior
        for _ in range(2):                                   # update size from the observed path so far
            _, proj = h_forecast(sigs, k, t, H, scales=AMP, cur_scale=A)
            ok = ~np.isnan(sigs[k]); s_ = proj[ok] - proj[0]; y_ = sigs[k][ok]
            lam = 0.25 * N                                     # prior worth a quarter of a cycle of data
            A = max((s_ @ y_ + lam * prior) / (s_ @ s_ + lam), 0.05)
        famp, _ = h_forecast(sigs, k, t, H, scales=AMP, cur_scale=A)

        # --- Hvol: scale by each cycle's own volatility; current cycle by trailing 365-day volatility
        fvol, _ = h_forecast(sigs, k, t, H, scales=VOLC, cur_scale=vol365[i])

        rows.append(dict(date=t, cycle=k + 1, i=i, real=fwd(t) if t + H * DAY <= last else np.nan,
                         H0=f0, Hamp=famp, Hvol=fvol, size_prior=prior, size_now=A, sig=vol[i]))
        t += 7 * DAY
    R = pd.DataFrame(rows)

    # --- quarterly and swing clocks: conditional mean of vol-scaled forward returns (or residuals)
    doq = np.array([(d - d.to_period("Q").start_time).days for d in dates])
    dsp = np.where(piv >= 0, np.arange(len(P)) - piv, -1)

    def clock_pred(target_col, use_resid):
        out_q, out_s = [], []
        for _, row in R.iterrows():
            known = R[(R.date + H * DAY) <= row.date]               # outcomes known at this date
            if use_resid:
                known = known.dropna(subset=["real"])
                z = ((known.real - known[target_col]) / (known.sig * np.sqrt(H))).values
                ki = known.i.values.astype(int)
            else:
                # raw version: all weekly dates since 2013 with known outcomes
                ki = np.arange(i_of(pd.Timestamp("2013-06-01")), row.i - H, 7)
                z = (lp[ki + H] - lp[ki]) / (vol[ki] * np.sqrt(H))
            if len(ki) < 20:
                out_q.append(0.0); out_s.append(0.0); continue
            i = int(row.i); sc = vol[i] * np.sqrt(H); zbar = z.mean()
            dq = np.abs(doq[ki] - doq[i]); dq = np.minimum(dq, 91 - dq)
            mq = dq <= 7
            out_q.append((z[mq].mean() - zbar) * sc if mq.sum() >= 5 else 0.0)
            ms = (mode[ki] == mode[i]) & (np.abs(dsp[ki] - dsp[i]) <= 21)
            out_s.append((z[ms].mean() - zbar) * sc if ms.sum() >= 5 else 0.0)
        return np.array(out_q), np.array(out_s)

    R["Q_raw"], R["S_raw"] = clock_pred(None, False)
    R["Q"], R["S"] = clock_pred("H0", True)
    R["Q_amp"], R["S_amp"] = clock_pred("Hamp", True)
    return R


def stack(R, base, comps, H, l2=1.0):
    """Combined = base + sum w_c * comp_c, weights fitted on earlier out-of-sample rows with known outcomes,
    ridge-shrunk to 0 and clipped at 0."""
    out = []
    W = []
    for j, row in R.iterrows():
        kn = R[(R.date + H * DAY <= row.date) & R.real.notna()]
        w = np.zeros(len(comps))
        if len(kn) >= 26:
            X = kn[comps].values; y = (kn.real - kn[base]).values
            s = X.std(0) + 1e-12
            w = np.linalg.solve((X / s).T @ (X / s) / len(kn) + l2 * np.eye(len(comps)), (X / s).T @ y / len(kn)) / s
            w = np.clip(w, 0, None)
        W.append(w); out.append(row[base] + row[comps].values @ w)
    return np.array(out), np.array(W)


def score(R, col, mask):
    d = R[mask & R.real.notna()]
    e = d[col] - d.real
    return dict(n=len(d), hit=float((np.sign(d[col]) == np.sign(d.real)).mean()),
                corr=float(np.corrcoef(d[col], d.real)[0, 1]) if d[col].std() > 0 else float("nan"),
                skill=float(1 - (e ** 2).mean() / (d.real ** 2).mean()))


if __name__ == "__main__":
    path = sys.argv[1]; outdir = sys.argv[2] if len(sys.argv) > 2 else "."
    rng = np.random.default_rng(0)
    report = {}
    for H in (90, 30):
        R = run(path, H)
        R["H0+QS"], Wa = stack(R, "H0", ["Q", "S"], H)
        R["Hamp+QS"], Wb = stack(R, "Hamp", ["Q_amp", "S_amp"], H)
        R["H0+S"], _ = stack(R, "H0", ["S"], H)
        R["H0+Q"], _ = stack(R, "H0", ["Q"], H)
        # shuffled control for the stacked clocks: permute Q,S in 13-week blocks, re-stack
        test = R.cycle >= 3
        ctrl = []
        for _ in range(100):
            Rc = R.copy(); n = len(Rc); blocks = np.array_split(np.arange(n), max(n // 13, 1))
            perm = np.concatenate([blocks[b] for b in rng.permutation(len(blocks))])
            Rc[["Q", "S"]] = R[["Q", "S"]].values[perm]
            c, _ = stack(Rc, "H0", ["Q", "S"], H); Rc["c"] = c
            ctrl.append(score(Rc, "c", test)["hit"])
        cols = ["H0", "Hamp", "Hvol", "Q_raw", "S_raw", "H0+Q", "H0+S", "H0+QS", "Hamp+QS"]
        res = {c: {f"cycle{cy}": score(R, c, R.cycle == cy) for cy in (3, 4)} | {"all": score(R, c, test)} for c in cols}
        up = float((R[test].real.dropna() > 0).mean())
        report[H] = dict(results=res, always_up=max(up, 1 - up), ctrl_hit_median=float(np.median(ctrl)),
                         ctrl_hit_95=float(np.quantile(ctrl, 0.95)),
                         final_weights={"H0+QS": dict(zip(["Q", "S"], np.round(Wa[-1], 3).tolist())),
                                        "Hamp+QS": dict(zip(["Q", "S"], np.round(Wb[-1], 3).tolist()))},
                         size_now=float(R.size_now.iloc[-1]), size_prior=float(R.size_prior.iloc[-1]))
        R.to_csv(f"{outdir}/multifilter_{H}d.csv", index=False)
        print(f"\n=== horizon {H} days | test = Cycles 3+4 model-made weekly calls ===")
        print(f"{'model':10s} {'n':>4s} {'hit':>6s} {'corr':>6s} {'skill':>7s} | {'C3 hit':>6s} {'C4 hit':>6s}")
        for c in cols:
            a, c3, c4 = res[c]["all"], res[c]["cycle3"], res[c]["cycle4"]
            print(f"{c:10s} {a['n']:4d} {a['hit']:6.1%} {a['corr']:6.2f} {a['skill']:7.2f} | {c3['hit']:6.1%} {c4['hit']:6.1%}")
        print(f"always-up {report[H]['always_up']:.1%} | shuffled-clock control (H0+QS): median {report[H]['ctrl_hit_median']:.1%}, 95th {report[H]['ctrl_hit_95']:.1%}")
        print("final stacking weights:", report[H]["final_weights"])
        print(f"cycle-4 size: prior {report[H]['size_prior']:.2f}, updated {report[H]['size_now']:.2f} (log run-up)")
    json.dump(report, open(f"{outdir}/multifilter_results.json", "w"), indent=1)
