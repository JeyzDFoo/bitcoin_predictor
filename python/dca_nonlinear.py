"""Non-linear weekly DCA weights vs linear, with a shuffled-signal control. Run: python dca_nonlinear.py btc.csv"""
import sys, json
import numpy as np, pandas as pd
import backtest as b

def sim(p, dates, s, buy_w, sell_f, base=100.0):
    coins = cash = spent = 0.0; eq = []
    for t, v in zip(dates, s):
        px = p.loc[t]; amt = base * buy_w(v); coins += amt / px; spent += amt
        f = sell_f(v)
        if f: q = coins * f; coins -= q; cash += q * px
        eq.append(coins * px + cash)
    eq = np.array(eq); return eq[-1] / spent, (eq / np.maximum.accumulate(eq) - 1).min(), spent

sg = lambda x: 1 / (1 + np.exp(-x))
RULES = {
 "plain DCA":                         (lambda v: 1, lambda v: 0),
 "linear buy + linear sell":          (lambda v: float(np.clip(1 + v / .3, 0, 2)), lambda v: float(np.clip(-v - .1, 0, .2) * .5)),
 "flat buy + step sell 5%":           (lambda v: 1, lambda v: .05 if v < -.1 else 0),
 "flat buy + convex sell (v^2)":      (lambda v: 1, lambda v: float(min(.25, 3 * max(0, -v - .05) ** 2 * 5))),
 "flat buy + sigmoid sell":           (lambda v: 1, lambda v: float(.15 * sg((-v - .15) / .05))),
 "sigmoid buy (0.3–1.7x) + step sell":(lambda v: .3 + 1.4 * sg(v / .1), lambda v: .05 if v < -.1 else 0),
 "convex buy (|v|^1.5) + step sell":  (lambda v: float(np.clip(1 + np.sign(v) * (abs(v) / .3) ** 1.5, 0, 2.5)), lambda v: .05 if v < -.1 else 0),
 "tail-only buy (2x if v>0.3) + step":(lambda v: 2 if v > .3 else 1, lambda v: .05 if v < -.1 else 0),
}
if __name__ == "__main__":
    p = b.load(sys.argv[1] if len(sys.argv) > 1 else b.URL)
    T = json.load(open("trend.json"))["taps"]["41"]; d = pd.to_datetime(T["d"], unit="ms"); s = np.array(T["p"]); k = d <= p.index[-1]; d, s = d[k], s[k]
    rng = np.random.default_rng(0); perms = [rng.permutation(s) for _ in range(300)]
    print(f"{'rule':38s} {'multiple':>8s} {'max dd':>7s} {'spent':>8s} | shuffle median / 95th pct")
    for n, (bw, sf) in RULES.items():
        m, dd, sp = sim(p, d, s, bw, sf); sh = np.array([sim(p, d, q, bw, sf)[0] for q in perms]) if n != "plain DCA" else np.array([m])
        print(f"{n:38s} {m:8.2f} {dd:7.0%} {sp:8,.0f} | {np.median(sh):.2f} / {np.percentile(sh,95):.2f}")
