#!/usr/bin/env bash
# One-time:  rustup target add wasm32-unknown-unknown
# Then:      ./build.sh   and   python3 -m http.server   -> open http://localhost:8000
set -euo pipefail
cd "$(dirname "$0")"
cargo build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/btc_fir.wasm fir.wasm
echo "Built fir.wasm ($(wc -c < fir.wasm) bytes). Serve this folder and open index.html."
