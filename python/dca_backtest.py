"""Weekly weighted DCA using the walk-forward 90d signal vs plain DCA. Run: python dca_backtest.py btc.csv"""
import sys, json
import numpy as np, pandas as pd
import backtest as b

def run(prices, dates, pred, buy_k, sell_thr, sell_frac, base=100.0):
    """each week: buy base*clip(1+buy_k*pred/0.3, 0, 2); if pred<-sell_thr sell sell_frac of holdings."""
    coins = cash = spent = 0.0; eq = []
    for t, s in zip(dates, pred):
        px = prices.loc[t]
        amt = base * float(np.clip(1 + buy_k * s / 0.3, 0, 2)) if buy_k else base
        coins += amt / px; spent += amt
        if sell_frac and s < -sell_thr:
            q = coins * sell_frac; coins -= q; cash += q * px
        eq.append(coins * px + cash)
    eq = np.array(eq); last = prices.loc[dates[-1]]
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    return dict(spent=spent, final=eq[-1], multiple=eq[-1] / spent, avg_cost=spent / (coins + cash / last) if coins + cash else np.nan, max_dd_of_equity=dd)

if __name__ == "__main__":
    p = b.load(sys.argv[1] if len(sys.argv) > 1 else b.URL)
    T = json.load(open("trend.json"))["taps"]["41"]
    dates = pd.to_datetime(T["d"], unit="ms"); pred = np.array(T["p"])
    dates = dates.where(dates <= p.index[-1]); keep = ~dates.isna(); dates, pred = dates[keep], pred[keep]
    rows = {"plain DCA": run(p, dates, pred, 0, 0, 0)}
    for k in (1, 2): rows[f"weighted buys k={k}"] = run(p, dates, pred, k, 0, 0)
    for k in (0, 1):
        for thr, fr in ((0.1, 0.05), (0.1, 0.10)):
            rows[f"{'weighted' if k else 'plain'} buys + sell {int(fr*100)}% when pred<-{thr}"] = run(p, dates, pred, k, thr, fr)
    # robustness: same rule with the signal shuffled in time (is the timing doing anything?)
    rng = np.random.default_rng(0); sh = []
    for _ in range(500): sh.append(run(p, dates, rng.permutation(pred), 1, 0.1, 0.05)["multiple"])
    r = pd.DataFrame(rows).T; pd.set_option("display.float_format", lambda v: f"{v:,.2f}", "display.width", 140)
    print(f"{len(dates)} weekly buys, {dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}\n"); print(r.to_string())
    m = rows["weighted buys + sell 5% when pred<-0.1"]["multiple"] if "weighted buys + sell 5% when pred<-0.1" in rows else None
    m = run(p, dates, pred, 1, 0.1, 0.05)["multiple"]
    print(f"\nShuffled-signal control (500 shuffles, same rule k=1, sell 5%): median {np.median(sh):.2f}x, 5–95%: {np.percentile(sh,5):.2f}–{np.percentile(sh,95):.2f}x vs real signal {m:.2f}x; real beats {np.mean(np.array(sh)<m)*100:.0f}% of shuffles")
