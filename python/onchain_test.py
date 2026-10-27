"""Does on-chain data (Coin Metrics community set) add information beyond the cycle signal? Walk-forward, purged, 90d.
Run: python onchain_test.py btc.csv"""
import sys, json
import numpy as np, pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
raw = pd.read_csv(path, parse_dates=["time"]).set_index("time").loc["2013":]
raw = raw[raw.PriceUSD.notna()]; raw = raw.reindex(pd.date_range(raw.index[0], raw.index[-1])).ffill()
P = raw.PriceUSD; lg = np.log
F = pd.DataFrame(index=raw.index)
F["mvrv"] = raw.CapMVRVCur
F["mvrv_30d_chg"] = raw.CapMVRVCur.diff(30)
F["hash_30d"] = lg(raw.HashRate).diff(30)
F["hash_vs_1y"] = lg(raw.HashRate) - lg(raw.HashRate).rolling(365).mean()
F["addr_30d"] = lg(raw.AdrActCnt.rolling(7).mean()).diff(30)
F["tx_30d"] = lg(raw.TxTfrCnt.rolling(7).mean()).diff(30)
F["netflow_30d"] = ((raw.FlowInExNtv - raw.FlowOutExNtv).rolling(30).sum() / raw.SplyCur)
F["issuance_usd_vs_1y"] = lg(raw.IssTotUSD.rolling(30).mean()) - lg(raw.IssTotUSD.rolling(365).mean())
F["mom_90d"] = lg(P).diff(90); F["dist_from_200d"] = lg(P) - lg(P.rolling(200).mean())
y = lg(P.shift(-90) / P)                                       # 90d forward log return
T = json.load(open("trend.json"))["taps"]["41"]; d = pd.to_datetime(T["d"], unit="ms")
cyc = pd.Series(T["p"], index=d); d = d[d + pd.Timedelta(days=90) <= P.index[-1]]; cyc = cyc.loc[d]
r = y.loc[d].values

# 1) what each feature says about the part of the return the cycle model misses
res = pd.Series(r - cyc.values, index=d)
print("Correlation with the cycle model's miss (r - pred), and with the raw 90d return, on the same dates:")
print(pd.DataFrame({"vs miss": F.loc[d].corrwith(res), "vs return": F.loc[d].corrwith(pd.Series(r, index=d))}).round(2).to_string())

# 2) walk-forward ridge on all weekly dates since 2014; target = 90d return, purged expanding fit
W = F.dropna().index[::7]; W = W[W + pd.Timedelta(days=90) <= P.index[-1]]
Fa = F.dropna(); X = Fa; Wset = W; Y = y.loc[Fa.index.intersection(y.dropna().index)]
def pred_at(t, cols, alpha=30.0):
    tr = Wset[Wset + pd.Timedelta(days=90) <= t]; m, s = X.loc[tr, cols].mean(), X.loc[tr, cols].std()
    A = ((X.loc[tr, cols] - m) / s).values; yy = Y.loc[tr].values; mu = yy.mean()
    w = np.linalg.solve(A.T @ A + alpha * np.eye(len(cols)), A.T @ (yy - mu)); return float((((X.loc[[t], cols] - m) / s).values @ w)[0] + mu)
sets = {"on-chain only": ["mvrv", "mvrv_30d_chg", "hash_30d", "hash_vs_1y", "addr_30d", "tx_30d", "netflow_30d", "issuance_usd_vs_1y"],
        "on-chain + price momentum": list(F.columns), "MVRV only": ["mvrv"]}
idx = pd.DatetimeIndex([t for t in d if t in Fa.index])
rr = y.loc[idx].values; cy = cyc.loc[idx].values
def sc(s):
    e = s - rr; return dict(corr=np.corrcoef(s, rr)[0, 1], skill=1 - (e ** 2).mean() / (rr ** 2).mean(), hit=(np.sign(s) == np.sign(rr)).mean())
rows = {"cycle model (current)": sc(cy)}
for n, c in sets.items():
    o = np.array([pred_at(t, c) for t in idx]); rows[n] = sc(o)
    rows[f"cycle + 0.5 x {n}"] = sc(cy + 0.5 * o); rows[f"average(cycle, {n})"] = sc(0.5 * cy + 0.5 * o)
pd.set_option("display.float_format", lambda v: f"{v:,.2f}", "display.width", 140)
print(f"\nWalk-forward, {len(idx)} weekly dates {idx[0]:%Y-%m-%d} to {idx[-1]:%Y-%m-%d}:"); print(pd.DataFrame(rows).T.to_string())
