"""
Does outside-market information (dollar, gold, tech stocks, volatility, yields, Fed liquidity) add to the
halving filter's 90-day call?  Pre-registered design, fixed before looking at results:

  * Base: H0 = current site model's walk-forward 90-day forecast (from multifilter_90d.csv).
  * Features at each weekly date t, using market data up to the day before t only:
      dxy_90   90-day log change in the dollar index        gold_90  90-day log change in gold
      ndx_90   90-day log change in the Nasdaq-100          semi_90  90-day log change in semiconductors (SOXX)
      vix_z    log VIX minus its trailing 1-year mean       real_90  90-day change in 10y real yield
      fed_13w  13-week log change in Fed total assets
  * Overlay: ridge regression of the halving filter's miss (realised - H0) on the standardised features,
    refit at every date on earlier dates whose 90-day outcome was already known (expanding window, from
    Cycle 2), heavily shrunk; combined call = H0 + overlay.  Also a macro-only model (no halving filter).
  * Score: direction right on Cycles 3+4 model-made weekly calls, vs H0, vs always-up, and vs the same
    overlay fitted on block-shuffled features (100 shuffles, 13-week blocks).
  * Single-feature results are reported for information only; the full set is the one that counts.

Run: python macro_test.py market.csv multifilter_90d.csv
"""
import sys, json
import numpy as np
import pandas as pd

H = 90
mk = pd.read_csv(sys.argv[1], parse_dates=["date"]).set_index("date").sort_index()
R = pd.read_csv(sys.argv[2], parse_dates=["date"])
mk = mk.reindex(pd.date_range(mk.index.min(), R.date.max())).ffill()
dxy = mk["dxy"].fillna(mk.get("dxy_broad"))

F = pd.DataFrame(index=mk.index)
lg = lambda s: np.log(s.where(s > 0))
F["dxy_90"] = lg(dxy).diff(90)
F["gold_90"] = lg(mk.gold).diff(90)
F["ndx_90"] = lg(mk.ndx).diff(90)
F["semi_90"] = lg(mk.soxx).diff(90)
F["vix_z"] = lg(mk.vix) - lg(mk.vix).rolling(365).mean()
F["real_90"] = mk.real10y.diff(90)
F["fed_13w"] = lg(mk.fed_assets).diff(91)
F = F.shift(1)                                        # only data from before the BTC date
FEATS = list(F.columns)

D = R.merge(F, left_on="date", right_index=True, how="left").dropna(subset=FEATS)
D = D.reset_index(drop=True)
test = (D.cycle >= 3) & D.real.notna()


def overlay(D, feats, target, l2=1.0, min_known=52):
    """Expanding-window ridge on standardised features; returns out-of-sample prediction per row."""
    out = np.zeros(len(D))
    X_all = D[feats].values
    for j in range(len(D)):
        kn = ((D.date + pd.Timedelta(days=H)) <= D.date.iloc[j]) & D.real.notna()
        if kn.sum() < min_known:
            continue
        X = X_all[kn.values]; y = target[kn.values]
        mu, sd = X.mean(0), X.std(0) + 1e-12
        Z = (X - mu) / sd
        w = np.linalg.solve(Z.T @ Z / len(y) + l2 * np.eye(len(feats)), Z.T @ (y - y.mean()) / len(y))
        out[j] = ((X_all[j] - mu) / sd) @ w      # no intercept: the overlay only shifts the call with the features
    return out


def hit(pred, mask=test):
    d = D[mask]; p = pred[mask.values]
    return float((np.sign(p) == np.sign(d.real.values)).mean())


resid = (D.real - D.H0).values
D["overlay"] = overlay(D, FEATS, resid)
D["H0+macro"] = D.H0 + D.overlay
D["macro_only"] = overlay(D, FEATS, D.real.values)

res = {}
for c in ["H0", "H0+macro", "macro_only"]:
    res[c] = {"all": hit(D[c].values), "cycle3": hit(D[c].values, test & (D.cycle == 3)), "cycle4": hit(D[c].values, test & (D.cycle == 4)),
              "corr": float(np.corrcoef(D.loc[test, c], D.loc[test, "real"])[0, 1])}

rng = np.random.default_rng(1)
ctrl = []
n = len(D); blocks = np.array_split(np.arange(n), max(n // 13, 1))
for _ in range(100):
    perm = np.concatenate([blocks[b] for b in rng.permutation(len(blocks))])
    Ds = D.copy(); Ds[FEATS] = D[FEATS].values[perm]
    ctrl.append(hit((D.H0 + overlay(Ds, FEATS, resid)).values))

single = {}
for f in FEATS:
    o = overlay(D, [f], resid)
    single[f] = {"H0+f": hit((D.H0 + o).values), "corr_with_miss": float(np.corrcoef(D.loc[test, f], resid[test.values])[0, 1]),
                 "corr_with_return": float(np.corrcoef(D.loc[test, f], D.loc[test, "real"])[0, 1])}

up = float((D.loc[test, "real"] > 0).mean())
out = dict(n=int(test.sum()), data_through=str(mk.index.max().date()), always_up=max(up, 1 - up), results=res,
           ctrl_median=float(np.median(ctrl)), ctrl_95=float(np.quantile(ctrl, 0.95)), single=single)
json.dump(out, open("macro_results.json", "w"), indent=1)
D[["date", "cycle", "real", "H0", "overlay", "H0+macro", "macro_only"] + FEATS].to_csv("macro_preds.csv", index=False)

print(f"n={out['n']} weekly 90-day calls, Cycles 3+4 | always-up {out['always_up']:.1%}")
for c, v in res.items():
    print(f"  {c:11s} hit {v['all']:.1%}  (C3 {v['cycle3']:.1%}, C4 {v['cycle4']:.1%})  corr {v['corr']:+.2f}")
print(f"  shuffled-feature control for H0+macro: median {out['ctrl_median']:.1%}, 95th {out['ctrl_95']:.1%}")
print("single features (information only):")
for f, v in single.items():
    print(f"  {f:8s} H0+f {v['H0+f']:.1%}   corr with filter's miss {v['corr_with_miss']:+.2f}   corr with BTC 90d return {v['corr_with_return']:+.2f}")
