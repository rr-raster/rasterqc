The goal of this project is to provides a high-performance Rust backend for computing statistics
 on raster imagery, exposed as a lightweight Python module via PyO3.
It is be designed to be called from python, for scientific workflows where large rasters, multi-band datasets, and custom per-pixel logic
 demand more speed than NumPy can reliably deliver. 
 
 By combining Rust's native performance, zero-cost abstractions, and safe memory model with Python's ease of use, the module offers fast, reliable raster analytics without changing your existing Python pipeline.

The Rust core implements efficient, parallelized routines for common raster statistics such as min, max, mean, standard deviation, histograms, and masked operations. It accepts NumPy arrays without copying, streams large rasters in chunks, and scales across multiple CPU cores. The result is a compact, low-overhead extension that significantly accelerates raster processing while remaining simple to install, import, and integrate into existing scientific image processing workflows.


Rough Performance benchmark estimate/comparison
On a 10,000 × 10,000 float32 raster:

|Method |  Time | Notes |
|----|---|---|
|NumPy	|~180–250 ms | Fast, but single‑threaded |
|Rust (single‑thread) | ~80–120 ms | LLVM auto‑vectorization |
|Rust (Rayon parallel) | ~20–40 ms | Fully multi‑core |
| Rust (SIMD tuned) | ~10–20 ms | Hand‑optimized |




# Getting Started
Environment and dependency management is done with pixi:

`pixi install`


# Compile the Rust:

`pixi run cargo build`


# Use in python example

`pixi run python examples/quickstart.py`


## Building & benchmarking

Use the provided scripts to build a release-quality extension and to make benchmarking reproducible across machines.

- Cargo build flags (local CPU tuning) are stored in `.cargo/config.toml`.
- Runtime benchmarking envs are provided in `.env.sample` — copy to `.env` and edit if needed.

Linux / WSL / macOS (bash):
```bash
# make .env from sample and tune threads
cp .env.sample .env
./scripts/build.sh
RAYON_NUM_THREADS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python examples/benchmark.py
```

Windows (PowerShell):
```powershell
# copy sample and build
Copy-Item .env.sample .env
.\scripts\build.ps1
$env:RAYON_NUM_THREADS='4'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
python .\examples\benchmark.py
```

Notes:
- Use `maturin develop --release` (the scripts do this) to ensure the Rust code is optimized.
- `.cargo/config.toml` with `target-cpu=native` gives best local performance; CI builds should avoid that flag if portability is required.

