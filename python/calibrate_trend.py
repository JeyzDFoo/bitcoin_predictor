"""
90-day trend track record for the page: walk-forward predicted vs actual log return
(weekly as-of dates in Cycles 3 and 4, nothing after the as-of date used for fitting).

Run: python calibrate_trend.py [btc.csv]  ->  trend.json
"""
import sys, json
import numpy as np, pandas as pd
import backtest as b
import model

H = 90
TAPS = [21, 41, 61, 81]
WARMUP = {2: 120, 3: 30}   # days of the current cycle observed before testing starts


def main(path):
    p = b.load(path); last = p.index[-1]; pc = model.clean(p, model.CLIP)
    out = {"horizon": H, "taps": {}}
    for taps in TAPS:
        D, P, R, A = [], [], [], []
        DAY = pd.Timedelta(days=1)
        lr = lambda a, c: float(np.log(p.loc[c] / p.loc[a]))
        for k in (2, 3):
            s, e = b.HAL[k], b.HAL[k + 1]
            t = s
            while t + pd.Timedelta(days=H) <= last and t < e:
                d0 = (t - s).days
                if d0 < WARMUP[k]:      # new cycle too young to fit: use the previous cycle's curve at the same age
                    pred, a = lr(b.HAL[k - 1] + d0 * DAY, b.HAL[k - 1] + (d0 + H) * DAY), 1
                else:
                    path_ = model.forecast_path(pc, k, t, taps)
                    f = lambda d: b.interp_at(path_, (t + pd.Timedelta(days=d) - s) / (e - s))
                    if t + pd.Timedelta(days=H) <= e:
                        pred, a = float(f(H) - f(0)), 0
                    else:               # crosses the halving: model up to it, then the cycle that just ended replays its first days
                        r = (t + pd.Timedelta(days=H) - e).days
                        pred, a = float(f((e - t).days) - f(0)) + lr(s, s + r * DAY), 1
                D.append(int(t.value // 1_000_000)); P.append(round(pred, 4)); A.append(a)
                R.append(round(lr(t, t + pd.Timedelta(days=H)), 4))
                t += pd.Timedelta(days=b.STEP)
        P_, R_ = np.array(P), np.array(R)
        hit = float((np.sign(P_) == np.sign(R_)).mean()); base = float(max((R_ > 0).mean(), (R_ < 0).mean()))
        up = float((R_ > 0).mean())
        out["taps"][str(taps)] = dict(d=D, p=P, r=R, a=A, hit=round(hit, 3), up=round(up, 3), n=len(D))
        A_ = np.array(A, bool)
        print(f"   model-only n={(~A_).sum()} hit={(np.sign(P_[~A_])==np.sign(R_[~A_])).mean():.3f} | prev-cycle-filled n={A_.sum()} hit={(np.sign(P_[A_])==np.sign(R_[A_])).mean():.3f}")
        strong = np.abs(P_) > 0.1
        print(f"taps {taps}: n={len(D)} hit={hit:.3f} always-up={up:.3f} | |pred|>0.1: n={strong.sum()} hit={(np.sign(P_[strong])==np.sign(R_[strong])).mean():.3f}")
    json.dump(out, open("trend.json", "w"), separators=(",", ":"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else b.URL)
