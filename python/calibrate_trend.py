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
        D, P, R = [], [], []
        for k in (2, 3):
            s, e = b.HAL[k], b.HAL[k + 1]
            t = b.HAL[k] + pd.Timedelta(days=WARMUP[k])
            while t + pd.Timedelta(days=H) <= min(last, e):
                path_ = model.forecast_path(pc, k, t, taps)
                f = lambda d: b.interp_at(path_, (t + pd.Timedelta(days=d) - s) / (e - s))
                D.append(int(t.value // 1_000_000)); P.append(round(float(f(H) - f(0)), 4))
                R.append(round(float(np.log(p.loc[t + pd.Timedelta(days=H)] / p.loc[t])), 4))
                t += pd.Timedelta(days=b.STEP)
        P_, R_ = np.array(P), np.array(R)
        hit = float((np.sign(P_) == np.sign(R_)).mean()); base = float(max((R_ > 0).mean(), (R_ < 0).mean()))
        up = float((R_ > 0).mean())
        out["taps"][str(taps)] = dict(d=D, p=P, r=R, hit=round(hit, 3), up=round(up, 3), n=len(D))
        strong = np.abs(P_) > 0.1
        print(f"taps {taps}: n={len(D)} hit={hit:.3f} always-up={up:.3f} | |pred|>0.1: n={strong.sum()} hit={(np.sign(P_[strong])==np.sign(R_[strong])).mean():.3f}")
    json.dump(out, open("trend.json", "w"), separators=(",", ":"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else b.URL)
