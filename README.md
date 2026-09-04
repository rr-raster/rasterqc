This project provides a high-performance Rust backend for computing statistics
 on raster imagery, exposed as a lightweight Python module via PyO3.
It will be designed for scientific workflows where large rasters, multi-band datasets, and custom per-pixel logic
 demand more speed than NumPy can reliably deliver. 
 
 By combining Rust's native performance, zero-cost abstractions, and safe memory model with Python's ease of use, the module offers fast, reliable raster analytics without changing your existing Python pipeline.

The Rust core implements efficient, parallelized routines for common raster statistics such as min, max, mean, standard deviation, histograms, and masked operations. It accepts NumPy arrays without copying, streams large rasters in chunks, and scales across multiple CPU cores. The result is a compact, low-overhead extension that significantly accelerates raster processing while remaining simple to install, import, and integrate into existing geospatial workflows.


Rough Performance benchmark estimate/comparison
On a 10,000 × 10,000 float32 raster:

|Method |  Time | Notes |
|----|---|---|
|NumPy	|~180–250 ms | Fast, but single‑threaded |
|Rust (single‑thread) | ~80–120 ms | LLVM auto‑vectorization |
|Rust (Rayon parallel) | ~20–40 ms | Fully multi‑core |
| Rust (SIMD tuned) | ~10–20 ms | Hand‑optimized |



* Repository Structure

rasterstats/
├── Cargo.toml
|── pixi.toml
├── pyproject.toml
├── README.md
├── src/
│   ├── lib.rs                # PyO3 module entry point
│   ├── stats.rs              # Core raster statistics engine (pure Rust)
│   ├── histogram.rs          # Optional histogram routines
│   ├── interop.rs            # NumPy <-> Rust zero-copy conversion
│   └── utils.rs              # Shared helpers (dtype dispatch, masks, etc.)
├── python/
│   ├── __init__.py           # Python package wrapper
│   └── api.py                # Thin Python API (optional)
└── tests/
    ├── python_tests.py       # pytest-based tests
    └── rust_tests.rs         # Rust unit tests for core logic

# Getting Started
