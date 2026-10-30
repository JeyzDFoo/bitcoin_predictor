"""Daily live record: today's 90-day call from the cycle filter, the frozen dollar adjustment, and the
combined call, appended to python/live_log.csv.  Outcomes are filled in once 90 days have passed.
Nothing here is refit on the outcomes, so this log is a true out-of-sample record from 2026-10-30 on.
Run (from repo root, in GitHub Actions): python python/live_track.py"""
import os, json
import numpy as np
import pandas as pd
from multifilter import clean_series, cycle_sig, h_forecast, HAL, DAY

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG = os.path.join(HERE, "live_log.csv")


def btc_series():
    D = json.load(open(os.path.join(ROOT, "data.json")))
    p = pd.Series(D["daily"]["p"], index=pd.date_range(D["daily"]["start"], periods=len(D["daily"]["p"])))
    import yfinance as yf
    y = yf.download("BTC-USD", start=str((p.index[-1] - 5 * DAY).date()), progress=False)["Close"].squeeze()
    y.index = pd.to_datetime(y.index).tz_localize(None).normalize()
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    y = y[(y.index > p.index[-1]) & (y.index < today)]          # completed UTC days only
    s = pd.concat([p, y]).sort_index()
    return s.reindex(pd.date_range(s.index[0], s.index[-1])).ffill()


def main():
    s = btc_series()
    t = s.index[-1]; t0 = s.index[0]
    CL = clean_series(s.values)
    k = int(np.searchsorted(HAL, t, side="right")) - 1
    sigs = [cycle_sig(CL, t0, j, t) for j in range(k + 1)]
    h0, _ = h_forecast(sigs, k, t, 90)

    mk = pd.read_csv(os.path.join(HERE, "market.csv"), parse_dates=["date"]).set_index("date")
    dx = mk["dxy"].fillna(mk["dxy_broad"]).reindex(pd.date_range(mk.index.min(), t)).ffill()
    dxy_90 = float(np.log(dx.loc[t - DAY]) - np.log(dx.loc[t - 91 * DAY]))
    P = json.load(open(os.path.join(HERE, "dollar.json")))
    adj = P["w"] * (dxy_90 - P["mu"]) / P["sd"]

    log = pd.read_csv(LOG, parse_dates=["date"]) if os.path.exists(LOG) else pd.DataFrame()
    row = dict(date=t, btc=round(float(s.loc[t]), 2), cycle_call=round(float(h0), 5), dxy_90=round(dxy_90, 5),
               dollar_adj=round(float(adj), 5), combined=round(float(h0 + adj), 5), outcome=np.nan,
               cycle_right=np.nan, combined_right=np.nan)
    if len(log) and (log.date == t).any():
        print("already logged", t.date())
    else:
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    # fill outcomes that are now known
    for i, r in log.iterrows():
        tt = r.date + 90 * DAY
        if pd.isna(r.outcome) and tt in s.index:
            o = float(np.log(s.loc[tt] / s.loc[r.date]))
            log.loc[i, ["outcome", "cycle_right", "combined_right"]] = [round(o, 5), int(np.sign(o) == np.sign(r.cycle_call)), int(np.sign(o) == np.sign(r.combined))]
    log.to_csv(LOG, index=False, date_format="%Y-%m-%d")
    pct = lambda v: f"{np.expm1(v) * 100:+.1f}%"
    print(f"{t.date()} BTC {row['btc']:,.0f} | cycle {pct(h0)} | DXY 90d {dxy_90 * 100:+.1f}% -> adj {pct(adj)} | combined {pct(h0 + adj)}")


if __name__ == "__main__":
    main()
