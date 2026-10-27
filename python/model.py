"""Default model used by the page: shock-clipped prices + recency-weighted training pairs."""
import numpy as np
import backtest as b
from shock_experiment import clean
from recency_experiment import wfit

GAMMA = 0.5   # weight multiplier per cycle back (1 = no recency weighting)
CLIP = 2.5    # shock clip threshold in rolling sigmas (None = off)


def forecast_path(pc, k, t, taps, gamma=GAMMA):
    """Filter-projected path for cycle k as of date t, fit on the (clipped) price series `pc`."""
    sigs = [b.cycle_signal(pc, j, t) for j in range(k + 1)]
    pairs = [(sigs[j], sigs[j + 1]) for j in range(k)]
    h, bb = wfit(pairs, [gamma ** (k - 1 - j) for j in range(k)], taps)
    return b.conv_matrix(sigs[k - 1], taps) @ h + bb
