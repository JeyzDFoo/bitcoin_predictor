"""
Alignment experiment: align cycles by (a) halving only, (b/c) halving + tops/bottoms landmarks.
Piecewise-linear time warp maps each cycle onto a common template phase before the FIR is learned.
 mode 'halving' : landmarks = halving -> next halving
 mode 'mix'     : + top and bottom, current cycle uses only landmarks CONFIRMED by date t (causal)
 mode 'oracle'  : + true top and bottom of the current cycle (uses the future; upper bound, Cycle 3 only)
"""
import numpy as np, pandas as pd
import backtest as b

N = b.N
DD, UP = 0.25, 0.35      # top confirmed after a 25% drop from the high; bottom after a 35% rise from the low
MIN_AFTER = 150          # landmark must be >= this many days after the previous landmark


def landmarks_complete(p, k):
    s, e = b.HAL[k], b.HAL[k + 1]; seg = p.loc[s:e]
    top = seg.idxmax(); bot = seg.loc[top:].idxmin()
    return top, bot


def landmarks_causal(p, k, t):
    """Top/bottom of cycle k that are confirmed using prices up to t only."""
    s = b.HAL[k]; seg = p.loc[s:t]; out = []
    cand = seg.loc[s + pd.Timedelta(days=MIN_AFTER):]
    if cand.empty: return out
    top = cand.idxmax()
    if seg.loc[top:].iloc[-1] <= (1 - DD) * seg.loc[top]:
        out.append(top)
        after = seg.loc[top + pd.Timedelta(days=MIN_AFTER):]
        if not after.empty:
            bot = after.idxmin()
            if seg.loc[bot:].iloc[-1] >= (1 + UP) * seg.loc[bot]: out.append(bot)
    return out


def warp(k, lm_times, phis):
    s, e = b.HAL[k], b.HAL[k + 1]
    xs = [0.0] + [(x - s) / pd.Timedelta(days=1) for x in lm_times] + [(e - s) / pd.Timedelta(days=1)]
    ys = [0.0] + list(phis[:len(lm_times)]) + [1.0]
    return lambda days: np.interp(days, xs, ys)


def signal(p, k, t, w):
    s, e = b.HAL[k], b.HAL[k + 1]
    seg = p.loc[s:min(e, t)]
    days = np.asarray((seg.index - s) / pd.Timedelta(days=1)); u = w(days)
    out = np.interp(np.linspace(0, 1, N), u, np.log(seg.values / seg.values[0]))
    out[np.linspace(0, 1, N) > u[-1]] = np.nan
    return out


def run(path, mode, H=90, warm=120, step=7):
    p = b.load(path); last = p.index[-1]; rows = []
    comp = {j: landmarks_complete(p, j) for j in range(3)}
    for k in (2, 3):
        if mode == 'oracle' and k == 3: continue
        s, e = b.HAL[k], b.HAL[k + 1]
        past = [j for j in range(k)]
        L = lambda j: (b.HAL[j + 1] - b.HAL[j]) / pd.Timedelta(days=1)
        fT = np.mean([(comp[j][0] - b.HAL[j]) / pd.Timedelta(days=1) / L(j) for j in past])
        fB = np.mean([(comp[j][1] - b.HAL[j]) / pd.Timedelta(days=1) / L(j) for j in past])
        t = s + pd.Timedelta(days=warm)
        while t + pd.Timedelta(days=H) <= min(last, e):
            sigs = []
            for j in past:
                w = warp(j, list(comp[j]), [fT, fB]) if mode != 'halving' else warp(j, [], [])
                sigs.append(signal(p, j, b.HAL[j + 1], w))
            if mode == 'halving': lm, ph = [], []
            elif mode == 'oracle': lm, ph = list(landmarks_complete(p, k)), [fT, fB]
            else: lm = landmarks_causal(p, k, t); ph = [fT, fB]
            w = warp(k, lm, ph); sigs.append(signal(p, k, t, w))
            h, bb = b.fit([(sigs[j], sigs[j + 1]) for j in range(k)], 41)
            path_ = b.conv_matrix(sigs[k - 1], 41) @ h + bb
            d0 = (t - s) / pd.Timedelta(days=1); d1 = d0 + H
            f = b.interp_at(path_, float(w(d1))) - b.interp_at(path_, float(w(d0)))
            rows.append(dict(t=t, cycle=k + 1, pred=f, real=np.log(p.loc[t + pd.Timedelta(days=H)] / p.loc[t]), nlm=len(lm)))
            t += pd.Timedelta(days=step)
    return pd.DataFrame(rows)


def score(g):
    e = g.pred - g.real; rw = np.sqrt((g.real ** 2).mean())
    return pd.Series(dict(n=len(g), skill=1 - (e ** 2).mean() / rw ** 2, corr=np.corrcoef(g.pred, g.real)[0, 1], hit=(np.sign(g.pred) == np.sign(g.real)).mean()))


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else b.URL
    pd.set_option("display.float_format", lambda v: f"{v:.3f}", "display.width", 120)
    for H in (60, 90):
        res = {m: run(path, m, H) for m in ("halving", "mix", "oracle")}
        print(f"\n=== horizon {H}d ===")
        for lab, sel in [("Cycle 3, all test dates", lambda d: d[d.cycle == 3]),
                         ("Cycle 3, dates where >=1 landmark confirmed", None),
                         ("Cycle 4, all test dates", lambda d: d[d.cycle == 4]),
                         ("Cycle 4, dates where >=1 landmark confirmed", None)]:
            rows = {}
            for m, d in res.items():
                if sel is None:
                    c = 3 if "Cycle 3" in lab else 4
                    keep = res['mix'][(res['mix'].cycle == c) & (res['mix'].nlm >= 1)].t
                    sub = d[(d.cycle == c) & d.t.isin(keep)]
                else: sub = sel(d)
                if len(sub) > 5: rows[m] = score(sub)
            print(f"\n{lab}"); print(pd.DataFrame(rows).T.to_string())
