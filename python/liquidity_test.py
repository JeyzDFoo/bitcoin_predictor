"""Does Fed net liquidity add information beyond the 90-day cycle signal?

Pre-registered primary hypothesis, tested once:
  The 90-day change in Fed net liquidity (WALCL - WTREGEN - RRPONTSYD)
  predicts the next 90-day BTC log return, beyond the cycle model.

Secondary, only after the primary block is printed:
  the 90-day change in the 10-year real yield (DFII10) and in the
  nominal broad dollar index (DTWEXBGS).

Protocol, same as onchain_test.py:
  weekly as-of dates 2020-05-11 to 2026-02-21 (trend.json taps 41),
  target = 90-day forward log return from the Coin Metrics price used by
  backtest.load. At date t, train only on rows whose 90-day outcome was
  already known. Standardize with that purged window's mean and sample
  standard deviation. Ridge alpha is 30, the onchain_test.py value, fixed
  before any of these results were looked at. It is not scaled by sample
  size, matching that script.

Compared on identical dates:
  (a) cycle model alone (trend.json taps 41)
  (b) liquidity feature alone
  (c) cycle + 0.5 * liquidity prediction
  (d) stacked ridge on [cycle prediction, liquidity feature]

Also: 500 shuffled-feature permutations (seed 0), a purged leave-one-cycle-out
check, and the dca_nonlinear sell rule (flat buy, sell 5% when signal < -0.1).

M2SL is loaded and lagged, and is not tested. It is not in the hypothesis.

Data files in data/fred/ are the FRED series. The FRED graph CSV endpoints
(graph/fcsv and graph/fredgraph.csv) returned 404 or 500, or hung, from this
environment on 2026-10-27, so the files were taken from the releases FRED
republishes and spot-checked against the FRED pages:
  WALCL   H.4.1 total assets less eliminations, Wednesday, millions
          (2026-10-21 = 6753604, the FRED print)
  WTREGEN H.4.1 Treasury General Account, Wednesday, millions
  RRPONTSYD NY Fed overnight reverse repo, Treasury amount accepted, billions
          (FRED's source for this series; 2022-12-30 = 2553.716)
  DFII10  H.15 10-year inflation-indexed yield (2026-10-23 = 2.66)
  DTWEXBGS H.10 nominal broad dollar index, daily (2006-01-02 = 101.4155)
  M2SL    H.6 M2, seasonally adjusted, billions, dated month-end
          (2026-08-31 = 23342.8)

Publication lags, applied to the observation date so a value is used only
once it was public. The as-of is the close of date t, the same convention
as using PriceUSD on t.
  WALCL, WTREGEN  next business day (H.4.1 is Thursday 4:30pm for Wednesday;
                  Friday when Thursday is a holiday). About one day.
  RRPONTSYD       same calendar day (NY Fed prints the operation by 1:15pm ET)
  DFII10          same business day (H.15 at 4:15pm ET)
  DTWEXBGS        the following Monday 4:15pm, or the next business day if
                  that Monday is a holiday (H.10 releases the previous
                  business week)
  M2SL            fourth Tuesday of the next month, or the next business day
                  if that Tuesday is a holiday (H.6). About three weeks
                  after month-end. Not a 21-day shift from the 1st, which
                  would be before the release.

Run: python liquidity_test.py [btc.csv]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import GoodFriday, HolidayCalendarFactory, USFederalHolidayCalendar

import backtest as b
from dca_nonlinear import sim

HERE = Path(__file__).resolve().parent
FRED = HERE / "data" / "fred"
ALPHA = 30.0          # onchain_test.py default; fixed before seeing these results
N_PERM = 500
SEED = 0
H = 90
TEST_START = pd.Timestamp("2020-05-11")
TEST_END = pd.Timestamp("2026-02-21")
CYCLE4 = pd.Timestamp("2024-04-20")
TAPS = "41"

FedCal = HolidayCalendarFactory("FedHolidays", USFederalHolidayCalendar, GoodFriday)
HOLIDAYS = set(pd.DatetimeIndex(FedCal().holidays("1990-01-01", "2030-01-01")).normalize())


def roll_to_business(t):
    t = pd.Timestamp(t).normalize()
    while t.day_of_week >= 5 or t in HOLIDAYS:
        t += pd.Timedelta(days=1)
    return t


def release_h41(obs):
    return roll_to_business(pd.Timestamp(obs) + pd.Timedelta(days=1))


def release_h10(obs):
    obs = pd.Timestamp(obs)
    monday = obs.normalize() - pd.Timedelta(days=int(obs.day_of_week))
    return roll_to_business(monday + pd.Timedelta(days=7))


def release_m2(obs):
    obs = pd.Timestamp(obs)
    y, m = obs.year, obs.month + 1
    if m == 13:
        y, m = y + 1, 1
    first = pd.Timestamp(year=y, month=m, day=1)
    fourth_tuesday = first + pd.Timedelta(days=(1 - int(first.day_of_week)) % 7 + 21)
    return roll_to_business(fourth_tuesday)


def release_same_day(obs):
    return pd.Timestamp(obs).normalize()


def load_fred(series_id):
    df = pd.read_csv(FRED / f"{series_id}.csv", parse_dates=["observation_date"])
    s = df.set_index("observation_date")[series_id].astype(float).sort_index()
    return s[~s.index.duplicated(keep="last")]


def publish(raw, release_fn):
    """Point-in-time daily level: value dated on its release day, then forward-filled."""
    raw = raw.sort_index()
    released = pd.Series(raw.to_numpy(), index=pd.DatetimeIndex([release_fn(t) for t in raw.index]))
    released = released.groupby(level=0).last().sort_index()
    daily = pd.date_range(released.index.min(), released.index.max(), freq="D")
    return released.reindex(daily).ffill()


def change_90(level):
    return level.diff(H)


def metrics(pred, real):
    pred = np.asarray(pred, float)
    real = np.asarray(real, float)
    m = np.isfinite(pred) & np.isfinite(real)
    pred, real = pred[m], real[m]
    err = pred - real
    corr = float(np.corrcoef(pred, real)[0, 1]) if pred.std() > 0 else np.nan
    return dict(n=int(len(real)), corr=corr,
                skill=float(1 - np.mean(err ** 2) / np.mean(real ** 2)),
                hit=float(np.mean(np.sign(pred) == np.sign(real))))


def wf_1d(train_dates, train_x, train_y, test_dates, test_x, alpha=ALPHA):
    """Purged expanding ridge, one feature. Train rows need outcome known by the test date."""
    train_dates = np.asarray(train_dates, dtype="datetime64[ns]")
    test_dates = np.asarray(pd.DatetimeIndex(test_dates), dtype="datetime64[ns]")
    x = np.asarray(train_x, float)
    y = np.asarray(train_y, float)
    xt = np.asarray(test_x, float)
    cx, cy = np.cumsum(x), np.cumsum(y)
    cx2, cxy = np.cumsum(x * x), np.cumsum(x * y)
    ks = np.searchsorted(train_dates, test_dates - np.timedelta64(H, "D"), side="right") - 1
    preds = np.full(len(test_dates), np.nan)
    for i, k in enumerate(ks):
        if k < 1:
            continue
        n = k + 1
        mx, my = cx[k] / n, cy[k] / n
        ss = cx2[k] - n * mx * mx
        if ss <= 1e-18:
            continue
        sd = np.sqrt(ss / (n - 1))
        w = ((cxy[k] - n * mx * my) / sd) / ((n - 1) + alpha)
        preds[i] = (xt[i] - mx) / sd * w + my
    return preds


def wf_2d(train_dates, train_c, train_l, train_y, test_dates, test_c, test_l, alpha=ALPHA):
    """Purged expanding ridge on two features. Same alpha on each coefficient."""
    train_dates = np.asarray(train_dates, dtype="datetime64[ns]")
    test_dates = np.asarray(pd.DatetimeIndex(test_dates), dtype="datetime64[ns]")
    c = np.asarray(train_c, float)
    l = np.asarray(train_l, float)
    y = np.asarray(train_y, float)
    tc = np.asarray(test_c, float)
    tl = np.asarray(test_l, float)
    cc, cl, cy = np.cumsum(c), np.cumsum(l), np.cumsum(y)
    cc2, cl2 = np.cumsum(c * c), np.cumsum(l * l)
    ccl, ccy, cly = np.cumsum(c * l), np.cumsum(c * y), np.cumsum(l * y)
    ks = np.searchsorted(train_dates, test_dates - np.timedelta64(H, "D"), side="right") - 1
    preds = np.full(len(test_dates), np.nan)
    for i, k in enumerate(ks):
        if k < 1:
            continue
        n = float(k + 1)
        mc, ml, my = cc[k] / n, cl[k] / n, cy[k] / n
        sc2 = (cc2[k] - n * mc * mc) / (n - 1)
        sl2 = (cl2[k] - n * ml * ml) / (n - 1)
        if sc2 <= 1e-18 or sl2 <= 1e-18:
            continue
        sc, sl = np.sqrt(sc2), np.sqrt(sl2)
        s12 = (ccl[k] - n * mc * ml) / (sc * sl)
        s1y = (ccy[k] - n * mc * my) / sc
        s2y = (cly[k] - n * ml * my) / sl
        a = (n - 1) + alpha
        det = a * a - s12 * s12
        if det <= 1e-12:
            continue
        w1 = (a * s1y - s12 * s2y) / det
        w2 = (a * s2y - s12 * s1y) / det
        preds[i] = (tc[i] - mc) / sc * w1 + (tl[i] - ml) / sl * w2 + my
    return preds


def check_ridge_matches_solve(dates, x, y, test_dates, test_x, preds):
    i = int(np.where(np.isfinite(preds))[0][-1])
    t = np.datetime64(pd.Timestamp(test_dates[i]), "ns")
    k = int(np.searchsorted(np.asarray(dates, dtype="datetime64[ns]"), t - np.timedelta64(H, "D"), side="right") - 1)
    tr_x = np.asarray(x[:k + 1], float)
    tr_y = np.asarray(y[:k + 1], float)
    m, s, mu = tr_x.mean(), tr_x.std(ddof=1), tr_y.mean()
    A = ((tr_x - m) / s)[:, None]
    w = np.linalg.solve(A.T @ A + ALPHA * np.eye(1), A.T @ (tr_y - mu))
    direct = float((test_x[i] - m) / s * w[0] + mu)
    if abs(direct - preds[i]) > 1e-8:
        raise AssertionError(f"1d ridge mismatch {direct} vs {preds[i]}")


def check_ridge2_matches_solve(dates, c, l, y, test_dates, tc, tl, preds):
    i = int(np.where(np.isfinite(preds))[0][-1])
    t = np.datetime64(pd.Timestamp(test_dates[i]), "ns")
    k = int(np.searchsorted(np.asarray(dates, dtype="datetime64[ns]"), t - np.timedelta64(H, "D"), side="right") - 1)
    tr_c = np.asarray(c[:k + 1], float)
    tr_l = np.asarray(l[:k + 1], float)
    tr_y = np.asarray(y[:k + 1], float)
    mc, ml, my = tr_c.mean(), tr_l.mean(), tr_y.mean()
    sc, sl = tr_c.std(ddof=1), tr_l.std(ddof=1)
    A = np.column_stack(((tr_c - mc) / sc, (tr_l - ml) / sl))
    w = np.linalg.solve(A.T @ A + ALPHA * np.eye(2), A.T @ (tr_y - my))
    z = np.array([(tc[i] - mc) / sc, (tl[i] - ml) / sl])
    direct = float(z @ w + my)
    if abs(direct - preds[i]) > 1e-8:
        raise AssertionError(f"2d ridge mismatch {direct} vs {preds[i]}")


def phase_corrs(pred, real, step=13):
    out = []
    pred = np.asarray(pred, float)
    real = np.asarray(real, float)
    for p in range(step):
        sl = slice(p, None, step)
        if np.sum(np.isfinite(pred[sl])) < 5 or np.nanstd(pred[sl]) == 0:
            continue
        out.append(float(np.corrcoef(pred[sl], real[sl])[0, 1]))
    return np.asarray(out)


def forward_return(prices, dates):
    dates = pd.DatetimeIndex(dates)
    ahead = dates + pd.Timedelta(days=H)
    ok = ahead.isin(prices.index) & dates.isin(prices.index)
    r = pd.Series(np.nan, index=dates)
    r.loc[ok] = np.log(prices.loc[ahead[ok]].to_numpy() / prices.loc[dates[ok]].to_numpy())
    return r


def weekly_history(feature, prices, anchor):
    """Mondays aligned with the trend dates, from the first valid feature week through the test end."""
    start = feature.first_valid_index()
    dates = []
    t = anchor
    while t >= start:
        dates.append(t)
        t -= pd.Timedelta(days=7)
    t = anchor + pd.Timedelta(days=7)
    while t <= TEST_END:
        dates.append(t)
        t += pd.Timedelta(days=7)
    dates = pd.DatetimeIndex(sorted(set(dates)))
    x = feature.reindex(dates)
    y = forward_return(prices, dates)
    m = x.notna() & y.notna()
    return dates[m], x[m].to_numpy(), y[m].to_numpy()


def run_block(name, feature, prices, cycle, long_dates, long_x, long_y, test_dates):
    """(a)-(d) on the dates where every column is defined, plus shuffle and leave-one-cycle-out."""
    y_test = forward_return(prices, test_dates)
    x_test = feature.reindex(test_dates)
    base = pd.DataFrame({"date": test_dates, "y": y_test.to_numpy(), "cycle": cycle.reindex(test_dates).to_numpy(),
                         "x": x_test.to_numpy()}).dropna()
    # Stacked training frame: trend weeks only (the cycle prediction does not exist before them).
    stack = base.copy()
    # Liquidity-alone training uses the long weekly history. Predictions are then read on `base` dates.
    liq = wf_1d(long_dates, long_x, long_y, base["date"], base["x"])
    stacked = wf_2d(stack["date"], stack["cycle"], stack["x"], stack["y"], base["date"], base["cycle"], base["x"])
    check_ridge_matches_solve(long_dates, long_x, long_y, base["date"], base["x"].to_numpy(), liq)
    check_ridge2_matches_solve(stack["date"], stack["cycle"], stack["x"], stack["y"],
                               base["date"], base["cycle"].to_numpy(), base["x"].to_numpy(), stacked)
    base["liq"] = liq
    base["combo"] = base["cycle"] + 0.5 * base["liq"]
    base["stack"] = stacked
    common = base.dropna().reset_index(drop=True)
    real = common["y"].to_numpy()
    rows = {
        "cycle model": metrics(common["cycle"], real),
        "liquidity alone": metrics(common["liq"], real),
        "cycle + 0.5 x liquidity": metrics(common["combo"], real),
        "stacked ridge": metrics(common["stack"], real),
    }
    # One permutation of the long weekly feature. Cycle predictions stay on their dates.
    rng = np.random.default_rng(SEED)
    x_by_date = pd.Series(long_x, index=pd.DatetimeIndex(long_dates))
    sh_liq, sh_combo, sh_stack = [], [], []
    sh_corr = []
    for _ in range(N_PERM):
        xp = pd.Series(rng.permutation(long_x), index=x_by_date.index)
        # Stacked frame must be re-read from the permuted long series, in date order.
        sx = xp.reindex(stack["date"]).to_numpy()
        bx = xp.reindex(common["date"]).to_numpy()
        p_liq = wf_1d(long_dates, xp.to_numpy(), long_y, common["date"], bx)
        p_stack = wf_2d(stack["date"], stack["cycle"], sx, stack["y"], common["date"], common["cycle"], bx)
        sh_liq.append(metrics(p_liq, real)["skill"])
        sh_combo.append(metrics(common["cycle"].to_numpy() + 0.5 * p_liq, real)["skill"])
        sh_stack.append(metrics(p_stack, real)["skill"])
        sh_corr.append(metrics(p_stack, real)["corr"])
    shuffle = {k: np.asarray(v, float) for k, v in
               {"liquidity alone": sh_liq, "cycle + 0.5 x liquidity": sh_combo, "stacked ridge": sh_stack, "stacked corr": sh_corr}.items()}

    loco = {}
    for which, lo, hi in (("cycle 3", TEST_START, CYCLE4), ("cycle 4", CYCLE4, pd.Timestamp("2028-04-17"))):
        hold = (common["date"] >= lo) & (common["date"] < hi)
        sub = common.loc[hold]
        if len(sub) == 0:
            continue
        # Training rows outside the held-out cycle. Purge is inside the walker.
        long_keep = ~((pd.DatetimeIndex(long_dates) >= lo) & (pd.DatetimeIndex(long_dates) < hi))
        stack_keep = ~((stack["date"] >= lo) & (stack["date"] < hi))
        p_liq = wf_1d(np.asarray(long_dates)[long_keep], long_x[long_keep], long_y[long_keep], sub["date"], sub["x"])
        if stack_keep.sum() >= 2:
            p_stack = wf_2d(stack.loc[stack_keep, "date"], stack.loc[stack_keep, "cycle"], stack.loc[stack_keep, "x"],
                            stack.loc[stack_keep, "y"], sub["date"], sub["cycle"], sub["x"])
        else:
            p_stack = np.full(len(sub), np.nan)
        yy = sub["y"].to_numpy()
        loco[which] = {
            "cycle model": metrics(sub["cycle"], yy),
            "liquidity alone": metrics(p_liq, yy) if np.isfinite(p_liq).any() else None,
            "cycle + 0.5 x liquidity": metrics(sub["cycle"].to_numpy() + 0.5 * p_liq, yy) if np.isfinite(p_liq).any() else None,
            "stacked ridge": metrics(p_stack, yy) if np.isfinite(p_stack).any() else None,
        }
    return dict(name=name, common=common, rows=rows, shuffle=shuffle, loco=loco, up=float((real > 0).mean()))


def print_block(res, primary=False):
    common = res["common"]
    real = common["y"].to_numpy()
    print(f"\n{res['name']}")
    print(f"identical dates: n={len(common)}  {common['date'].iloc[0]:%Y-%m-%d} to {common['date'].iloc[-1]:%Y-%m-%d}")
    print(f"always-up hit rate: {res['up']:.3f}")
    tbl = pd.DataFrame(res["rows"]).T
    print(tbl.to_string(float_format=lambda v: f"{v:,.3f}"))
    miss = real - common["cycle"].to_numpy()
    x = common["x"].to_numpy()
    print(f"descriptive, full common sample, not the test: corr(feature, return)={np.corrcoef(x, real)[0, 1]:+.3f}  "
          f"corr(feature, cycle miss)={np.corrcoef(x, miss)[0, 1]:+.3f}")
    print(f"shuffled feature, {N_PERM} perms, seed {SEED} (share of null skill >= real; null median; null 95th)")
    for key in ("liquidity alone", "cycle + 0.5 x liquidity", "stacked ridge"):
        null = res["shuffle"][key]
        real_s = res["rows"][key]["skill"]
        share = float(np.mean(null >= real_s))
        print(f"  {key:28s}  skill {real_s:+.3f}   null>={share:.3f}   median {np.nanmedian(null):+.3f}   p95 {np.nanpercentile(null, 95):+.3f}")
    null_c = res["shuffle"]["stacked corr"]
    real_c = res["rows"]["stacked ridge"]["corr"]
    print(f"  {'stacked corr':28s}  corr  {real_c:+.3f}   null>={float(np.mean(null_c >= real_c)):.3f}   "
          f"median {np.nanmedian(null_c):+.3f}   p95 {np.nanpercentile(null_c, 95):+.3f}")
    print("leave-one-cycle-out (purged; training rows from the held-out cycle are excluded)")
    for which, models in res["loco"].items():
        print(f"  {which}")
        for model, met in models.items():
            if met is None:
                print(f"    {model:28s}  not identified")
            else:
                print(f"    {model:28s}  n={met['n']:3d}  corr={met['corr']:+.3f}  skill={met['skill']:+.3f}  hit={met['hit']:.3f}")
    n = len(common)
    print(f"overlap caveat: {n} weekly 90-day returns share most of their window. "
          f"A rough independent count is n*7/90 = {n * 7 / 90:.1f}. "
          f"Correlation t-stats that treat the weeks as independent are not valid.")
    for label, col in (("cycle model", "cycle"), ("stacked ridge", "stack")):
        pc = phase_corrs(common[col], real)
        if len(pc):
            print(f"  non-overlapping corr, {label}: min {pc.min():+.3f}  median {np.median(pc):+.3f}  max {pc.max():+.3f}  ({len(pc)} phases)")
    if primary:
        cycle_s = res["rows"]["cycle model"]["skill"]
        stack_s = res["rows"]["stacked ridge"]["skill"]
        share = float(np.mean(res["shuffle"]["stacked ridge"] >= stack_s))
        holds = stack_s > cycle_s and share <= 0.05
        print()
        if holds:
            print("PRIMARY RESULT: the hypothesis holds on the pre-registered test. "
                  f"Stacked skill {stack_s:+.3f} is above the cycle model {cycle_s:+.3f}, "
                  f"and only {share:.1%} of shuffled-liquidity fits match or beat it.")
        else:
            print("PRIMARY RESULT: the hypothesis does not hold. "
                  "The 90-day change in Fed net liquidity does not add out-of-sample information beyond the cycle model.")
            print(f"  stacked skill {stack_s:+.3f} versus cycle skill {cycle_s:+.3f}; "
                  f"share of shuffled fits with skill at least this high = {share:.3f} "
                  f"(the bar was stacked skill higher than the cycle model and that share <= 0.05).")
        c3 = res["loco"].get("cycle 3", {}).get("stacked ridge")
        c4 = res["loco"].get("cycle 4", {}).get("stacked ridge")
        c4_cycle = res["loco"].get("cycle 4", {}).get("cycle model")
        if c3 is None:
            print("  Leave-cycle-3-out stacked ridge is not identified: the cycle prediction starts in cycle 3, so holding that cycle out leaves no stacked training rows.")
        if c4 is not None and c4_cycle is not None:
            print(f"  Leave-cycle-4-out stacked skill {c4['skill']:+.3f} versus cycle skill on those dates {c4_cycle['skill']:+.3f}.")
    return holds if primary else None


def dca_table(prices, common):
    rule = (lambda v: 1.0, lambda v: 0.05 if v < -0.1 else 0.0)
    plain = (lambda v: 1.0, lambda v: 0.0)
    dates = pd.DatetimeIndex(common["date"])
    print("\nDCA on the same dates: flat buy, sell 5% of coins when the signal is below -0.1 (dca_nonlinear.py).")
    print(f"{'signal':28s} {'multiple':>8s} {'max dd':>8s} {'spent':>8s}")
    rows = [("plain DCA (no sell)", None)]
    for label, col in (("cycle model", "cycle"), ("liquidity alone", "liq"),
                       ("cycle + 0.5 x liquidity", "combo"), ("stacked ridge", "stack")):
        rows.append((label, common[col].to_numpy()))
    # plain uses a zero signal only so the sell never fires; buy is flat either way
    for label, sig in rows:
        bw, sf = plain if sig is None else rule
        signal = np.zeros(len(dates)) if sig is None else sig
        multiple, dd, spent = sim(prices, dates, signal, bw, sf)
        print(f"{label:28s} {multiple:8.2f} {dd:8.0%} {spent:8,.0f}")


def timing_asserts(raw):
    w, tga = raw["WALCL"], raw["WTREGEN"]
    wal = publish(w, release_h41)
    # Wednesday 2020-05-06 is released Thursday 2020-05-07. It is not known on the Wednesday.
    assert wal.loc["2020-05-06"] == w.loc["2020-04-29"]
    assert wal.loc["2020-05-07"] == w.loc["2020-05-06"]
    assert wal.loc["2020-05-11"] == w.loc["2020-05-06"]
    assert publish(tga, release_h41).loc["2020-05-11"] == tga.loc["2020-05-06"]
    dxy = publish(raw["DTWEXBGS"], release_h10)
    # Monday 2020-05-11 releases the week ending Friday 2020-05-08. Friday itself was not yet public.
    assert dxy.loc["2020-05-08"] == raw["DTWEXBGS"].loc["2020-05-01"]
    assert dxy.loc["2020-05-11"] == raw["DTWEXBGS"].loc["2020-05-08"]
    yld = publish(raw["DFII10"], release_same_day)
    assert yld.loc["2020-05-11"] == raw["DFII10"].loc["2020-05-11"]
    rrp = publish(raw["RRPONTSYD"], release_same_day)
    assert abs(rrp.loc["2022-12-30"] - raw["RRPONTSYD"].loc["2022-12-30"]) < 1e-6
    assert release_m2(pd.Timestamp("2019-08-31")) == pd.Timestamp("2019-09-24")
    assert release_m2(pd.Timestamp("2026-08-31")) == pd.Timestamp("2026-09-22")


def net_liquidity(raw):
    wal = publish(raw["WALCL"], release_h41)
    tga = publish(raw["WTREGEN"], release_h41)
    rrp = publish(raw["RRPONTSYD"], release_same_day) * 1000.0  # billions -> millions
    idx = wal.index.union(tga.index).union(rrp.index)
    net = wal.reindex(idx).ffill() - tga.reindex(idx).ffill() - rrp.reindex(idx).ffill()
    net = net.dropna()
    return change_90(net)


def main(path):
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    raw = {k: load_fred(k) for k in ("WALCL", "WTREGEN", "RRPONTSYD", "DFII10", "DTWEXBGS", "M2SL")}
    timing_asserts(raw)
    print("Macro series, observation range, publication rule")
    print(f"  WALCL     {raw['WALCL'].index.min():%Y-%m-%d} to {raw['WALCL'].index.max():%Y-%m-%d}  millions, H.4.1 next business day  (2026-10-21 = {raw['WALCL'].loc['2026-10-21']:,.0f})")
    print(f"  WTREGEN   {raw['WTREGEN'].index.min():%Y-%m-%d} to {raw['WTREGEN'].index.max():%Y-%m-%d}  millions, H.4.1 next business day")
    print(f"  RRPONTSYD {raw['RRPONTSYD'].index.min():%Y-%m-%d} to {raw['RRPONTSYD'].index.max():%Y-%m-%d}  billions, same day  (2022-12-30 = {raw['RRPONTSYD'].loc['2022-12-30']:.3f})")
    print(f"  DFII10    {raw['DFII10'].index.min():%Y-%m-%d} to {raw['DFII10'].index.max():%Y-%m-%d}  percent, H.15 same day  (2026-10-23 = {raw['DFII10'].loc['2026-10-23']:.2f})")
    print(f"  DTWEXBGS  {raw['DTWEXBGS'].index.min():%Y-%m-%d} to {raw['DTWEXBGS'].index.max():%Y-%m-%d}  index, H.10 following Monday  (2006-01-02 = {raw['DTWEXBGS'].iloc[0]:.4f})")
    print(f"  M2SL      {raw['M2SL'].index.min():%Y-%m-%d} to {raw['M2SL'].index.max():%Y-%m-%d}  billions, H.6 fourth Tuesday of the next month  (2026-08-31 = {raw['M2SL'].iloc[-1]:,.1f})")
    print("  M2SL is not in the pre-registered hypothesis and is not tested.")
    print(f"Ridge alpha = {ALPHA:g} (fixed; onchain_test.py). Permutations = {N_PERM}, seed {SEED}. No parameter was chosen on the test dates.")

    prices = b.load(path)
    T = json.load(open(HERE / "trend.json"))["taps"][TAPS]
    trend_dates = pd.to_datetime(T["d"], unit="ms")
    cycle = pd.Series(T["p"], index=trend_dates).sort_index()
    test_dates = cycle.index[(cycle.index >= TEST_START) & (cycle.index <= TEST_END)]
    # Realised target has to exist. The cycle signal itself is the published one.
    y_all = forward_return(prices, test_dates)
    test_dates = test_dates[y_all.notna()]
    print(f"\nPrices {prices.index.min():%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}. "
          f"Cycle signal taps {TAPS}: {len(test_dates)} weeks {test_dates.min():%Y-%m-%d} to {test_dates.max():%Y-%m-%d} with a realised {H}-day return.")

    liq = net_liquidity(raw)
    print(f"Fed net liquidity is WALCL - WTREGEN - 1000*RRPONTSYD, millions, each series lagged, then the 90-day change.")
    print(f"  90-day change available {liq.first_valid_index():%Y-%m-%d} to {liq.last_valid_index():%Y-%m-%d}.")

    long_dates, long_x, long_y = weekly_history(liq, prices, TEST_START)
    # The halving Saturday is a trend date and has to be in the frame so it can be trained on later.
    extra = test_dates.difference(pd.DatetimeIndex(long_dates))
    if len(extra):
        x_e = liq.reindex(extra)
        y_e = forward_return(prices, extra)
        m = x_e.notna() & y_e.notna()
        long_dates = pd.DatetimeIndex(long_dates).append(extra[m]).sort_values()
        long_x = liq.reindex(long_dates).to_numpy()
        long_y = forward_return(prices, long_dates).to_numpy()
    print("\n===== PRIMARY: 90-day change in Fed net liquidity (WALCL - WTREGEN - RRPONTSYD) =====")
    primary = run_block("Fed net liquidity, 90-day change", liq, prices, cycle, long_dates, long_x, long_y, test_dates)
    print_block(primary, primary=True)
    dca_table(prices, primary["common"])
    sys.stdout.flush()

    print("\n===== SECONDARY (run after the primary block above; not used to change it) =====")
    for series_id, release, label in (
        ("DFII10", release_same_day, "DFII10, 90-day change in the real yield"),
        ("DTWEXBGS", release_h10, "DTWEXBGS, 90-day change in the broad dollar"),
    ):
        feat = change_90(publish(raw[series_id], release))
        d, x, y = weekly_history(feat, prices, TEST_START)
        extra = test_dates.difference(pd.DatetimeIndex(d))
        if len(extra):
            x_e = feat.reindex(extra)
            y_e = forward_return(prices, extra)
            m = x_e.notna() & y_e.notna()
            d = pd.DatetimeIndex(d).append(extra[m]).sort_values()
            x = feat.reindex(d).to_numpy()
            y = forward_return(prices, d).to_numpy()
        res = run_block(label, feat, prices, cycle, d, x, y, test_dates)
        print_block(res, primary=False)
        cyc_s = res["rows"]["cycle model"]["skill"]
        stk_s = res["rows"]["stacked ridge"]["skill"]
        share = float(np.mean(res["shuffle"]["stacked ridge"] >= stk_s))
        if stk_s > cyc_s and share <= 0.05:
            print(f"SECONDARY {series_id}: stacked skill is above the cycle model and outside the shuffle null.")
        else:
            print(f"SECONDARY {series_id}: does not add out-of-sample information beyond the cycle model "
                  f"(stacked skill {stk_s:+.3f} vs cycle {cyc_s:+.3f}; shuffle share {share:.3f}).")
        sys.stdout.flush()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else b.URL)
