//! BTC halving cycles as a learned FIR filter, trained by gradient descent (Adam).
//! No dependencies. Exposes a tiny C ABI so the browser can drive it directly.
//!
//! Memory layout (all f64):
//!   sig_ptr()   -> 4 * N   cycle signals C1..C4 (NaN = not yet observed), written by JS
//!   h_ptr()     -> MAXT    learned taps (first `taps` entries valid)
//!   proj_ptr()  -> N       projected Cycle 4 path, anchored at last observed point
//!   stats_ptr() -> 8       [val_rmse, in_rmse, gain, centroid_taps, i_last, taps, 0, 0]

pub const N: usize = 200;
pub const MAXT: usize = 101;

static mut SIG: [f64; 4 * N] = [f64::NAN; 4 * N];
static mut H: [f64; MAXT] = [0.0; MAXT];
static mut PROJ: [f64; N] = [0.0; N];
static mut STATS: [f64; 8] = [0.0; 8];

#[no_mangle]
pub extern "C" fn n_points() -> u32 {
    N as u32
}
#[no_mangle]
pub extern "C" fn sig_ptr() -> *mut f64 {
    &raw mut SIG as *mut f64
}
#[no_mangle]
pub extern "C" fn h_ptr() -> *const f64 {
    &raw const H as *const f64
}
#[no_mangle]
pub extern "C" fn proj_ptr() -> *const f64 {
    &raw const PROJ as *const f64
}
#[no_mangle]
pub extern "C" fn stats_ptr() -> *const f64 {
    &raw const STATS as *const f64
}

/// Row `n` of the centred, edge-padded convolution matrix for input `x`.
#[inline]
fn conv_row(x: &[f64], taps: usize, n: usize, out: &mut [f64]) {
    let half = (taps / 2) as isize;
    for j in 0..taps {
        let idx = (n as isize + half - j as isize).clamp(0, N as isize - 1) as usize;
        out[j] = x[idx];
    }
}

struct Fit {
    h: Vec<f64>,
    b: f64,
    rmse: f64,
}

/// Train h, b by full-batch Adam on (input, target) cycle pairs.
fn train(pairs: &[(&[f64], &[f64])], taps: usize, epochs: usize, l2: f64, smooth: f64) -> Fit {
    // Build design matrix (rows with a known target only).
    let mut a: Vec<f64> = Vec::new();
    let mut y: Vec<f64> = Vec::new();
    let mut row = vec![0.0; taps];
    for (x, t) in pairs {
        for n in 0..N {
            if t[n].is_nan() {
                continue;
            }
            conv_row(x, taps, n, &mut row);
            a.extend_from_slice(&row);
            y.push(t[n]);
        }
    }
    let m = y.len();
    let half = taps / 2;

    // Tapered penalty: outer taps cost more (windowing).
    let w: Vec<f64> = (0..taps)
        .map(|i| {
            let r = if half == 0 { 0.0 } else { (i as f64 - half as f64).abs() / half as f64 };
            l2 * (1.0 + 200.0 * r.powi(4))
        })
        .collect();

    let mut h = vec![0.0; taps];
    h[half] = 0.5;
    let mut b = 0.0;
    let (mut mh, mut vh) = (vec![0.0; taps], vec![0.0; taps]);
    let (mut mb, mut vb) = (0.0, 0.0);
    let (lr, b1, b2, eps) = (0.05, 0.9_f64, 0.999_f64, 1e-8);
    let mut err = vec![0.0; m];
    let mut gh = vec![0.0; taps];

    for t in 1..=epochs {
        // forward
        for r in 0..m {
            let ar = &a[r * taps..(r + 1) * taps];
            let mut s = b;
            for j in 0..taps {
                s += ar[j] * h[j];
            }
            err[r] = s - y[r];
        }
        // gradients
        for j in 0..taps {
            gh[j] = w[j] * h[j];
        }
        let inv_m = 1.0 / m as f64;
        for r in 0..m {
            let e = err[r] * inv_m;
            let ar = &a[r * taps..(r + 1) * taps];
            for j in 0..taps {
                gh[j] += ar[j] * e;
            }
        }
        if taps >= 3 {
            for i in 0..taps - 2 {
                let d = smooth * (h[i] - 2.0 * h[i + 1] + h[i + 2]);
                gh[i] += d;
                gh[i + 1] -= 2.0 * d;
                gh[i + 2] += d;
            }
        }
        let gb = err.iter().sum::<f64>() * inv_m;
        // Adam
        let (c1, c2) = (1.0 - b1.powi(t as i32), 1.0 - b2.powi(t as i32));
        for j in 0..taps {
            mh[j] = b1 * mh[j] + (1.0 - b1) * gh[j];
            vh[j] = b2 * vh[j] + (1.0 - b2) * gh[j] * gh[j];
            h[j] -= lr * (mh[j] / c1) / ((vh[j] / c2).sqrt() + eps);
        }
        mb = b1 * mb + (1.0 - b1) * gb;
        vb = b2 * vb + (1.0 - b2) * gb * gb;
        b -= lr * (mb / c1) / ((vb / c2).sqrt() + eps);
    }

    let mut se = 0.0;
    for r in 0..m {
        let ar = &a[r * taps..(r + 1) * taps];
        let mut s = b;
        for j in 0..taps {
            s += ar[j] * h[j];
        }
        se += (s - y[r]).powi(2);
    }
    Fit { h, b, rmse: (se / m as f64).sqrt() }
}

fn apply(x: &[f64], h: &[f64], b: f64, out: &mut [f64]) {
    let taps = h.len();
    let mut row = vec![0.0; taps];
    for n in 0..N {
        conv_row(x, taps, n, &mut row);
        out[n] = b + row.iter().zip(h).map(|(a, c)| a * c).sum::<f64>();
    }
}

/// Fit the filter with `taps` taps (forced odd, 3..=MAXT) and project Cycle 4.
#[no_mangle]
pub extern "C" fn run(taps: u32, epochs: u32, l2: f64, smooth: f64) {
    let mut taps = (taps as usize).clamp(3, MAXT);
    if taps % 2 == 0 {
        taps += 1;
    }
    let epochs = epochs as usize;
    let sig: &[f64] = unsafe { &*(&raw const SIG as *const [f64; 4 * N]) };
    let (c1, c2, c3, c4) = (&sig[0..N], &sig[N..2 * N], &sig[2 * N..3 * N], &sig[3 * N..4 * N]);

    let i_last = (0..N).rev().find(|&i| !c4[i].is_nan()).unwrap_or(0);

    // Out-of-sample check: fit on C1->C2, C2->C3 only, score on observed C4.
    let v = train(&[(c1, c2), (c2, c3)], taps, epochs, l2, smooth);
    let mut vp = vec![0.0; N];
    apply(c3, &v.h, v.b, &mut vp);
    let (mut se, mut cnt) = (0.0, 0.0);
    for i in 0..N {
        if !c4[i].is_nan() {
            se += (vp[i] - c4[i]).powi(2);
            cnt += 1.0;
        }
    }
    let val_rmse = (se / cnt).sqrt();

    // Final fit including the observed part of C4.
    let f = train(&[(c1, c2), (c2, c3), (c3, c4)], taps, epochs, l2, smooth);
    let mut proj = vec![0.0; N];
    apply(c3, &f.h, f.b, &mut proj);
    let shift = c4[i_last] - proj[i_last];

    let gain: f64 = f.h.iter().sum();
    let half = (taps / 2) as f64;
    let centroid = f.h.iter().enumerate().map(|(i, v)| (i as f64 - half) * v).sum::<f64>() / gain;

    unsafe {
        let hp = &raw mut H as *mut f64;
        for i in 0..MAXT {
            *hp.add(i) = if i < taps { f.h[i] } else { 0.0 };
        }
        let pp = &raw mut PROJ as *mut f64;
        for i in 0..N {
            *pp.add(i) = proj[i] + shift;
        }
        let sp = &raw mut STATS as *mut f64;
        *sp.add(0) = val_rmse;
        *sp.add(1) = f.rmse;
        *sp.add(2) = gain;
        *sp.add(3) = centroid;
        *sp.add(4) = i_last as f64;
        *sp.add(5) = taps as f64;
    }
}
