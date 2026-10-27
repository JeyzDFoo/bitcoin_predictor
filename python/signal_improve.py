"""Candidate signal improvements, all scored on the same weekly dates (90d, walk-forward signals) + the simple sell-rule DCA.
Run: python signal_improve.py btc.csv"""
import sys, json
import numpy as np, pandas as pd
import backtest as b
from dca_nonlinear import sim

p = b.load(sys.argv[1] if len(sys.argv) > 1 else b.URL); H = 90; DAY = pd.Timedelta(days=1)
T = json.load(open("trend.json"))["taps"]; d = pd.to_datetime(T["41"]["d"], unit="ms"); R = np.array(T["41"]["r"])
S = {f"FIR {t}": np.array(T[t]["p"]) for t in T}
lr = lambda a, c: float(np.log(p.loc[a] / p.loc[c])) * -1
def analog(t):
    k = max(i for i in range(4) if b.HAL[i] <= t); d0 = (t - b.HAL[k]).days
    return float(np.mean([lr(b.HAL[j] + d0 * DAY, b.HAL[j] + (d0 + H) * DAY) for j in range(k)]))
A = np.array([analog(t) for t in d])
S["analog (earlier cycles, same age)"] = A
S["ensemble: mean of FIR 21/41/61/81"] = np.mean([S[f"FIR {t}"] for t in T], axis=0)
S["ensemble + analog 50/50"] = 0.5 * S["ensemble: mean of FIR 21/41/61/81"] + 0.5 * A
S["FIR 41 shrunk x0.5"] = 0.5 * S["FIR 41"]
def sign_only(x): return np.sign(x) * 0.12
S["FIR 41 sign only (±0.12)"] = sign_only(S["FIR 41"])
S["ensemble, sign-agreement only"] = np.where(np.sign(S["FIR 21"]) == np.sign(S["FIR 81"]), S["ensemble: mean of FIR 21/41/61/81"], 0)
k = d <= p.index[-1]
rows = {}
step = lambda v: .05 if v < -.1 else 0
for n, s in S.items():
    s = s[k]; r = R[k]; e = s - r; m = np.abs(s) > 1e-9
    mult, dd, _ = sim(p, d[k], s, lambda v: 1, step)
    rows[n] = dict(corr=np.corrcoef(s, r)[0, 1], skill=1 - (e ** 2).mean() / (r ** 2).mean(), hit=(np.sign(s[m]) == np.sign(r[m])).mean(), DCA_multiple=mult, DCA_maxdd=dd)
pd.set_option("display.float_format", lambda v: f"{v:,.2f}", "display.width", 140)
print(pd.DataFrame(rows).T.to_string())
