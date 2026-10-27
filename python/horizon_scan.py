"""Same walk-forward cycle model scored at shorter horizons, daily as-of dates (model-only dates, no halving fill).
Run: python horizon_scan.py btc.csv"""
import sys
import numpy as np, pandas as pd
import backtest as b, model

p = b.load(sys.argv[1] if len(sys.argv) > 1 else b.URL); pc = model.clean(p, model.CLIP); last = p.index[-1]
HS = [1, 3, 7, 14, 30, 60, 90]; WARM = {2: 120, 3: 30}; DAY = pd.Timedelta(days=1)
rows = []
for k in (2, 3):
    s, e = b.HAL[k], b.HAL[k + 1]; t = s + WARM[k] * DAY
    while t + 90 * DAY <= min(last, e):
        path = model.forecast_path(pc, k, t, 41); f = lambda x: b.interp_at(path, (t + x * DAY - s) / (e - s))
        rows.append(dict(t=t, **{f"p{h}": float(f(h) - f(0)) for h in HS}, **{f"r{h}": float(np.log(p.loc[t + h * DAY] / p.loc[t])) for h in HS}))
        t += DAY
D = pd.DataFrame(rows)
out = {}
for h in HS:
    P, R = D[f"p{h}"].values, D[f"r{h}"].values; e = P - R
    out[f"{h}d"] = dict(n=len(D), corr=np.corrcoef(P, R)[0, 1], skill=1 - (e ** 2).mean() / (R ** 2).mean(), hit=(np.sign(P) == np.sign(R)).mean(), always_up=max((R > 0).mean(), (R < 0).mean()),
                        typical_move=np.abs(R).mean(), typical_pred=np.abs(P).mean())
pd.set_option("display.float_format", lambda v: f"{v:,.3f}", "display.width", 140); print(pd.DataFrame(out).T.to_string())
D.to_csv("horizon_scan.csv", index=False)
