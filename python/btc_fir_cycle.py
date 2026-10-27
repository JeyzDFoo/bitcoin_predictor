"""
BTC halving cycles as a signal-processing problem
-------------------------------------------------
Each halving cycle is a signal: log(price / price at halving), sampled on a
common phase grid (0 = halving, 1 = next halving).

A two-sided FIR filter h is learned by gradient descent so that

    cycle[k+1][n]  ~=  sum_j h[j] * cycle[k][n - j]  +  b

i.e. the filter maps one cycle's shape onto the next: taps off-centre shift
timing (earlier/later peaks), total gain sets amplitude (diminishing returns).

Training pairs: C1->C2, C2->C3, and C3->C4 on the part of C4 already observed.
The filter is then run over C3 to project the rest of C4.

Run:  pip install numpy pandas matplotlib
      python btc_fir_cycle.py
"""
import numpy as np
import pandas as pd

HALVINGS = ["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20", "2028-04-17"]
DATA_URL = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"

N = 200          # phase samples per cycle (~1 week each)
TAPS = 41        # filter length (odd, centred) -> covers +/- ~20 weeks of shift
LR = 0.05        # gradient descent step
EPOCHS = 20000
L2 = 1e-3        # ridge on taps
SMOOTH = 1e-1    # second-difference penalty -> smooth impulse response


# ---------------- data ----------------
def load_prices(path=DATA_URL):
    d = pd.read_csv(path, usecols=["time", "PriceUSD"], parse_dates=["time"])
    return d.dropna().set_index("time")["PriceUSD"]


def cycle_signal(prices, start, end):
    """log(P / P_halving) on the phase grid; NaN where not yet observed."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    seg = prices.loc[start:end]
    phase_obs = (seg.index - start) / (end - start)
    grid = np.linspace(0, 1, N)
    sig = np.interp(grid, phase_obs, np.log(seg.values / seg.iloc[0]))
    sig[grid > phase_obs[-1]] = np.nan
    return sig, seg.iloc[0]


# ---------------- FIR model ----------------
def conv_matrix(x):
    """Rows = output samples, cols = taps; edge-padded, centred."""
    half = TAPS // 2
    xp = np.pad(x, half, mode="edge")
    return np.stack([xp[n:n + TAPS][::-1] for n in range(N)])


def train(pairs, log_every=5000):
    A = np.vstack([conv_matrix(x) for x, _ in pairs])
    y = np.concatenate([t for _, t in pairs])
    ok = ~np.isnan(y)
    A, y = A[ok], y[ok]
    m = len(y)

    D = np.diff(np.eye(TAPS), 2, axis=0)            # second-difference operator
    r = np.abs(np.arange(TAPS) - TAPS // 2) / (TAPS // 2)
    W = L2 * (1 + 200 * r**4)                         # tapered window: edge taps cost more
    h = np.zeros(TAPS); h[TAPS // 2] = 0.5           # start as a damped identity
    b = 0.0
    # Adam
    mh, vh, mb, vb = np.zeros(TAPS), np.zeros(TAPS), 0.0, 0.0
    for t in range(1, EPOCHS + 1):
        err = A @ h + b - y
        loss = 0.5 * np.mean(err**2) + 0.5 * np.sum(W * h**2) + 0.5 * SMOOTH * np.sum((D @ h)**2)
        gh = A.T @ err / m + W * h + SMOOTH * D.T @ (D @ h)
        gb = err.mean()
        mh = 0.9 * mh + 0.1 * gh; vh = 0.999 * vh + 0.001 * gh**2
        mb = 0.9 * mb + 0.1 * gb; vb = 0.999 * vb + 0.001 * gb**2
        h -= LR * (mh / (1 - 0.9**t)) / (np.sqrt(vh / (1 - 0.999**t)) + 1e-8)
        b -= LR * (mb / (1 - 0.9**t)) / (np.sqrt(vb / (1 - 0.999**t)) + 1e-8)
        if log_every and (t % log_every == 0 or t == 1):
            print(f"   epoch {t:>6}  loss {loss:.5f}")
    rmse = np.sqrt(np.mean((A @ h + b - y)**2))
    return h, b, rmse


def apply(h, b, x):
    return conv_matrix(x) @ h + b


# ---------------- run ----------------
def main(prices, plot_path="btc_fir_projection.png"):
    h_dates = [pd.Timestamp(d) for d in HALVINGS]
    cycles, base = [], []
    for k in range(4):
        s, p0 = cycle_signal(prices, h_dates[k], h_dates[k + 1])
        cycles.append(s); base.append(p0)
    C1, C2, C3, C4 = cycles
    known = ~np.isnan(C4)
    grid = np.linspace(0, 1, N)

    print("=" * 62)
    print(" BTC HALVING CYCLES — LEARNED FIR FILTER")
    print("=" * 62)
    print(f" Data through {prices.index[-1]:%Y-%m-%d} | Cycle 4 observed: {grid[known][-1]:.1%}")

    # 1) Out-of-sample check: train on C1->C2, C2->C3 only, predict observed C4
    print("\n Validation fit (C1->C2, C2->C3):")
    h_v, b_v, _ = train([(C1, C2), (C2, C3)], log_every=10000)
    val = apply(h_v, b_v, C3)
    val_rmse = np.sqrt(np.mean((val[known] - C4[known])**2))
    print(f"   out-of-sample error on observed C4: {val_rmse:.3f} log  (~{(np.exp(val_rmse)-1)*100:.0f}%)")

    # 2) Final fit including the observed part of C4
    print("\n Final fit (adds observed C3->C4):")
    h, b, rmse = train([(C1, C2), (C2, C3), (C3, C4)])
    proj = apply(h, b, C3)

    # Anchor the projection to the last observed point so the path is continuous
    i_last = np.where(known)[0][-1]
    proj_anch = proj + (C4[i_last] - proj[i_last])

    gain = h.sum()
    centroid = (np.arange(TAPS) - TAPS // 2) @ h / h.sum()
    shift_days = centroid / N * (h_dates[4] - h_dates[3]).days

    to_price = lambda v: base[3] * np.exp(v)
    fut = slice(i_last, N)
    i_peak = i_last + np.argmax(proj_anch[fut])
    i_low = i_last + np.argmin(proj_anch[fut])
    date_at = lambda i: h_dates[3] + (h_dates[4] - h_dates[3]) * grid[i]

    print("\n Filter")
    print(f"   gain (sum of taps)      {gain:.2f}   -> each cycle ~{gain:.0%} the amplitude of the last")
    print(f"   timing shift            {shift_days:+.0f} days   ({'later' if shift_days > 0 else 'earlier'} than prior cycle)")
    print(f"   in-sample error         {rmse:.3f} log")
    print("\n Projected Cycle 4 path")
    print(f"   last observed           {date_at(i_last):%Y-%m-%d}  ${to_price(C4[i_last]):>10,.0f}")
    print(f"   projected high          {date_at(i_peak):%Y-%m-%d}  ${to_price(proj_anch[i_peak]):>10,.0f}")
    print(f"   projected low           {date_at(i_low):%Y-%m-%d}  ${to_price(proj_anch[i_low]):>10,.0f}")
    print(f"   at next halving         {h_dates[4]:%Y-%m-%d}  ${to_price(proj_anch[-1]):>10,.0f}")
    print(f"   band (± val. error)     ${to_price(proj_anch[-1]-val_rmse):,.0f} – ${to_price(proj_anch[-1]+val_rmse):,.0f}")
    print("=" * 62)
    print(" Pattern-matching on 3 cycles; not a forecast or advice.")
    print("=" * 62)

    # ---------- plot ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 1]})
    for k, (c, col) in enumerate(zip([C1, C2, C3], ["#c9c9c9", "#a8a8a8", "#7a7a7a"])):
        ax.plot(grid, c, color=col, lw=1, label=f"Cycle {k+1}")
    ax.plot(grid[known], C4[known], color="#1f6feb", lw=2.2, label="Cycle 4 (actual)")
    ax.plot(grid[i_last:], proj_anch[i_last:], color="#1f6feb", lw=2.2, ls="--", label="Cycle 4 (projected)")
    ax.fill_between(grid[i_last:], proj_anch[i_last:] - val_rmse, proj_anch[i_last:] + val_rmse,
                    color="#1f6feb", alpha=0.12, lw=0)
    ax.set_ylim(-1.5, 7.5)
    ax.set_xlabel("Phase (0 = halving, 1 = next halving)")
    ax.set_ylabel("log(price / price at halving)")
    ax.set_title("Halving cycles with FIR projection")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)
    lag = (np.arange(TAPS) - TAPS // 2) / N * (h_dates[4] - h_dates[3]).days
    ax2.stem(lag, h, basefmt=" ")
    ax2.set_xlabel("Tap offset (days)")
    ax2.set_title("Learned impulse response")
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    print(f" Chart saved: {plot_path}")


if __name__ == "__main__":
    main(load_prices())
