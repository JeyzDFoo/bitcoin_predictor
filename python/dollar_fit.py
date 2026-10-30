"""Freeze the dollar adjustment once (pre-registered for live tracking).
adjustment (log 90-day return) = w * (dxy_90 - mu) / sd, where dxy_90 = 90-day log change in DXY up to the day
before; w = ridge (penalty 1 on standardised scale) fit of the halving filter's 90-day miss on dxy_90,
using every walk-forward date with a known outcome (Cycles 2-4).  Run once: python dollar_fit.py"""
import json
import numpy as np
import pandas as pd
D = pd.read_csv("macro_preds.csv", parse_dates=["date"]).dropna(subset=["real", "dxy_90"])
x = D.dxy_90.values; y = (D.real - D.H0).values
mu, sd = x.mean(), x.std()
z = (x - mu) / sd
w = float((z @ (y - y.mean()) / len(y)) / (1 + 1.0))
out = dict(mu=float(mu), sd=float(sd), w=w, n=int(len(D)), fitted_on=f"{D.date.min():%Y-%m-%d} to {D.date.max():%Y-%m-%d}",
           frozen="2026-10-30", note="adj = w*(dxy_90-mu)/sd in log-return units; dxy_90 = ln DXY(t-1) - ln DXY(t-91)")
json.dump(out, open("dollar.json", "w"), indent=1)
print(out)
print("adj per +1 sd dollar move: %.1f%%" % (100 * np.expm1(w)), "| 1 sd of dxy_90 = %.1f%%" % (100 * sd))
