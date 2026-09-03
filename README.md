This project provides a high-performance Rust backend for computing statistics
 on raster imagery, exposed as a lightweight Python module via PyO3.
It will be designed for scientific workflows where large rasters, multi-band datasets, and custom per-pixel logic
 demand more speed than NumPy can reliably deliver. 
 
 By combining Rust's native performance, zero-cost abstractions, and safe memory model with Python's ease of use, the module offers fast, reliable raster analytics without changing your existing Python pipeline.

The Rust core implements efficient, parallelized routines for common raster statistics such as min, max, mean, standard deviation, histograms, and masked operations. It accepts NumPy arrays without copying, streams large rasters in chunks, and scales across multiple CPU cores. The result is a compact, low-overhead extension that significantly accelerates raster processing while remaining simple to install, import, and integrate into existing geospatial workflows.


Rough Performance benchmark estimate/comparison
On a 10,000 × 10,000 float32 raster:

|Method |  Time | Notes |
|NumPy	|~180–250 ms | Fast, but single‑threaded |
|Rust (single‑thread) | ~80–120 ms | LLVM auto‑vectorization |
|Rust (Rayon parallel) | ~20–40 ms | Fully multi‑core |
| Rust (SIMD tuned) | ~10–20 ms | Hand‑optimized |