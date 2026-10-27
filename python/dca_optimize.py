"""Walk-forward optimisation of the DCA rule: each January, pick rule parameters using ONLY earlier weeks, trade the next year with them.
Run: python dca_optimize.py btc.csv"""
import sys, json, itertools
import numpy as np, pandas as pd
import backtest as b

GRID = [dict(thr=t, frac=f, k=k, redeploy=r) for t, f, k, r in itertools.product((.05, .1, .15, .2), (.03, .05, .1, .15), (0, 1), (0, .25))]

def step(st, px, s, P, base=100.0):
    amt = base * (float(np.clip(1 + P["k"] * s / .3, 0, 2)) if P["k"] else 1)
    if P["redeploy"] and s > .1 and st["cash"] > 0: amt += st["cash"] * P["redeploy"]; st["cash"] *= 1 - P["redeploy"]
    st["coins"] += amt / px; st["spent"] += base * (float(np.clip(1 + P["k"] * s / .3, 0, 2)) if P["k"] else 1)
    if s < -P["thr"]: q = st["coins"] * P["frac"]; st["coins"] -= q; st["cash"] += q * px
    return st["coins"] * px + st["cash"]

def run(p, d, s, params_for):
    st = dict(coins=0., cash=0., spent=0.); eq = [step(st, p.loc[t], v, params_for(t)) for t, v in zip(d, s)]
    eq = np.array(eq); return eq, st

def score(eq, st):
    return (eq[-1] / st["spent"]) * (1 + (eq / np.maximum.accumulate(eq) - 1).min())   # multiple, penalised by worst drawdown

if __name__ == "__main__":
    p = b.load(sys.argv[1] if len(sys.argv) > 1 else b.URL)
    T = json.load(open("trend.json"))["taps"]["41"]; d = pd.to_datetime(T["d"], unit="ms"); s = np.array(T["p"]); k = d <= p.index[-1]; d, s = d[k], s[k]
    plain = dict(thr=9, frac=0, k=0, redeploy=0)
    chosen = {}
    chosen[2020] = plain
    for yr in range(2021, 2027):
        cut = pd.Timestamp(f"{yr}-01-01"); m = d < cut
        if m.sum() < 30: chosen[yr] = plain; continue
        best = max(GRID, key=lambda P: score(*run(p, d[m], s[m], lambda t: P))); chosen[yr] = best
    wf = lambda t: chosen[t.year]
    out = {}
    for name, f in {"plain DCA": lambda t: plain, "walk-forward optimised (re-pick every Jan)": wf,
                    "fixed: sell 5% < -0.1 (hand-picked)": lambda t: dict(thr=.1, frac=.05, k=0, redeploy=0)}.items():
        eq, st = run(p, d, s, f); out[name] = dict(multiple=eq[-1] / st["spent"], max_dd=(eq / np.maximum.accumulate(eq) - 1).min(), spent=st["spent"])
    # also the in-sample "best ever" for contrast (this is the number that would mislead)
    bestP = max(GRID, key=lambda P: score(*run(p, d, s, lambda t: P))); eq, st = run(p, d, s, lambda t: bestP)
    out["IN-SAMPLE best (hindsight, not tradable)"] = dict(multiple=eq[-1] / st["spent"], max_dd=(eq / np.maximum.accumulate(eq) - 1).min(), spent=st["spent"])
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}", "display.width", 140); print(pd.DataFrame(out).T.to_string())
    print("\nparameters picked each January:"); [print(f"  {y}: " + ", ".join(f"{a}={b_}" for a, b_ in P.items())) for y, P in chosen.items()]
    print("\nin-sample best:", bestP)
