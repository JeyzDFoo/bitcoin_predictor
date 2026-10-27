// Native sanity check: cargo run --release --example check -- sig.txt 41
use std::{env, fs};

fn main() {
    let args: Vec<String> = env::args().collect();
    let text = fs::read_to_string(&args[1]).unwrap();
    let taps: u32 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(41);
    let sig = unsafe { std::slice::from_raw_parts_mut(btc_fir::sig_ptr(), 4 * btc_fir::N) };
    for (k, line) in text.lines().enumerate() {
        for (i, v) in line.split_whitespace().enumerate() {
            sig[k * btc_fir::N + i] = v.parse().unwrap();
        }
    }
    btc_fir::run(taps, 5000, 1e-3, 1e-1, 3);
    let st = unsafe { std::slice::from_raw_parts(btc_fir::stats_ptr(), 8) };
    let proj = unsafe { std::slice::from_raw_parts(btc_fir::proj_ptr(), btc_fir::N) };
    println!("val_rmse={:.4} in_rmse={:.4} gain={:.3} centroid_taps={:.3} i_last={}", st[0], st[1], st[2], st[3], st[4]);
    println!("final log value at halving = {:.4} -> ${:.0}", proj[btc_fir::N - 1], 64907.99 * proj[btc_fir::N - 1].exp());
}
