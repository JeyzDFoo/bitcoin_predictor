"""
Empirical uncertainty bands for the FIR forecast.

Walk-forward (same setup as backtest.py): for every weekly test date in Cycles 3 and 4,
refit using only past data, forecast the path, and record  error(h) = realised - forecast
log return over h days, for h = 0, 5, ... 60.  The quantiles of these errors, per horizon
and per tap count, are what the page draws as bands around the projection.

Run: python calibrate_bands.py [btc.csv]   ->  bands.json
"""
import sys, json
import numpy as np, pandas as pd
import backtest as b
import model

HS = list(range(0, 61, 5))
QS = {"q05": 5, "q25": 25, "q75": 75, "q95": 95}
TAPS = [21, 41, 61, 81]


def main(path):
    p = b.load(path); last = p.index[-1]; pc = model.clean(p, model.CLIP)
    errs = {t: {h: [] for h in HS} for t in TAPS}
    for k in (2, 3):
        s, e = b.HAL[k], b.HAL[k + 1]
        t = b.HAL[k] + pd.Timedelta(days=b.MIN_OBS_DAYS)
        while t + pd.Timedelta(days=max(HS)) <= min(last, e):
            for taps in TAPS:
                path_ = model.forecast_path(pc, k, t, taps)
                f0 = b.interp_at(path_, (t - s) / (e - s))
                for h in HS:
                    td = t + pd.Timedelta(days=h)
                    pred = b.interp_at(path_, (td - s) / (e - s)) - f0
                    real = np.log(p.loc[td] / p.loc[t])
                    errs[taps][h].append(real - pred)
            t += pd.Timedelta(days=b.STEP)
    out = {"horizons": HS, "n": len(errs[TAPS[0]][0]), "taps": {}}
    for taps in TAPS:
        out["taps"][str(taps)] = {q: [round(float(np.percentile(errs[taps][h], v)), 4) if h else 0.0 for h in HS]
                                  for q, v in QS.items()}
    json.dump(out, open("bands.json", "w"), separators=(",", ":"))
    print("test dates:", out["n"])
    for taps in (41,):
        for q in QS: print(taps, q, out["taps"][str(taps)][q])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else b.URL)
