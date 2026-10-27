# bitcoin_predictor

Halving cycles as signals. A learned FIR filter, trained by gradient descent, maps each cycle onto the next and projects the rest of the current one.

- `src/lib.rs` – Rust model (Adam, no dependencies), compiled to WebAssembly
- `index.html` – live page with a dial for the number of taps (falls back to a JS copy of the engine if `fir.wasm` is missing)
- `index.template.html` + `data.json` – page source and cycle data
- `python/btc_fir_cycle.py` – reference implementation and chart

## Run
```
rustup target add wasm32-unknown-unknown
./build.sh
python3 -m http.server   # open http://localhost:8000
```

Data: Coin Metrics open dataset. Pattern-matching on three cycles, not a forecast or investment advice.
