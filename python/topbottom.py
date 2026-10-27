"""Can we time cycle tops/bottoms? Leave-one-cycle-out: timing from earlier cycles only. Run: python topbottom.py btc.csv"""
import sys
import numpy as np, pandas as pd
import backtest as b, model

p = b.load(sys.argv[1] if len(sys.argv) > 1 else b.URL); last = p.index[-1]; DAY = pd.Timedelta(days=1)
H = b.HAL; rows = []
for k in range(4):
    s, e = H[k], min(H[k + 1], last); seg = p.loc[s:min(e, s + 600 * DAY)]; top = seg.idxmax()
    after = p.loc[top:min(top + 600 * DAY, last)]; bot = after.idxmin() if len(after) > 30 else None
    rows.append(dict(cycle=k + 1, top=top.date(), top_px=round(seg.max()), days_halving_to_top=(top - s).days, bottom=bot.date() if bot is not None else None,
                     bottom_px=round(after.min()) if bot is not None else None, days_top_to_bottom=(bot - top).days if bot is not None else None, drawdown=round(after.min() / seg.max() - 1, 2) if bot is not None else None))
T = pd.DataFrame(rows); pd.set_option("display.width", 160); print(T.to_string(index=False))
print(f"\ndata through {last.date()}; a 'bottom' in the last cycle is only provisional if data is recent.\n")
print("TOP TIMING, leave-one-cycle-out (days after halving):")
d = T.days_halving_to_top.values
for k in (1, 2, 3):
    pr = d[:k].mean(); print(f"  cycle {k+1}: predicted {pr:.0f} (from {k} earlier), actual {d[k]}, error {d[k]-pr:+.0f} days; naive spread of earlier cycles {d[:k].min()}-{d[:k].max()}")
print("\nFIR-model projected peak vs actual (cycle k projected from cycle k-1 using pairs before it):")
pc = model.clean(p, model.CLIP)
for k in (2, 3):
    e = H[k + 1]; t = min(last, e) if H[k + 1] > last else H[k + 1]
    sigs = [b.cycle_signal(pc, j, e if k < 3 else last) for j in range(k + 1)]
    from recency_experiment import wfit
    pairs = [(sigs[j], sigs[j + 1]) for j in range(k - 1)]; h, bb = wfit(pairs, [0.5 ** (k - 2 - j) for j in range(k - 1)], 41)
    path = b.conv_matrix(sigs[k - 1], 41) @ h + bb; ln = (H[k + 1] - H[k]).days; g = np.linspace(0, 1, 200) * ln; m = g <= 600; ph = g[m][path[m].argmax()] / ln
    print(f"  cycle {k+1}: model peak at {ph*ln:.0f} days after halving, actual {T.days_halving_to_top[k]}")
