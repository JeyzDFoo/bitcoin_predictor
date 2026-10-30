"""Download daily market data (non-price-of-BTC inputs) into python/market.csv.
Runs in GitHub Actions (open internet). Yahoo via yfinance for market prices, FRED for macro series.
Run: python fetch_market.py [out.csv]"""
import sys
import pandas as pd
import yfinance as yf

out = sys.argv[1] if len(sys.argv) > 1 else "python/market.csv"
YAHOO = {"dxy": "DX-Y.NYB", "gold": "GC=F", "ndx": "^NDX", "spx": "^GSPC", "vix": "^VIX", "soxx": "SOXX", "us10y": "^TNX"}
FRED = {"dxy_broad": "DTWEXBGS", "fed_assets": "WALCL", "real10y": "DFII10", "curve_10y2y": "T10Y2Y"}

cols = []
for name, tk in YAHOO.items():
    try:
        s = yf.download(tk, start="2012-01-01", progress=False, auto_adjust=False)["Close"].squeeze()
        s.name = name; cols.append(s); print(name, len(s), s.index[-1].date())
    except Exception as e:
        print("yahoo failed", name, e)
for name, sid in FRED.items():
    try:
        f = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", na_values=".")
        s = f.set_index(pd.to_datetime(f.iloc[:, 0])).iloc[:, 1].loc["2012":]
        s.name = name; cols.append(s); print(name, len(s), s.index[-1].date())
    except Exception as e:
        print("fred failed", name, e)

df = pd.concat(cols, axis=1).sort_index()
df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
df.index.name = "date"
df.round(4).to_csv(out)
print("wrote", out, df.shape)
