# rasterqc

Fast, parallelised raster statistics and QC metrics for scientific imaging.

## Features
- mean, median, std, skew, kurtosis
- percentiles
- histograms
- SNR
- uniformity
- dynamic range

## Installation
cargo add rasterqc

## Example
use rasterqc::stats::compute_stats;

let data = vec![1.0, 2.0, 3.0];
let s = compute_stats(&data).unwrap();
println!("mean = {}", s.mean);

## Documentation
https://docs.rs/rasterqc

## License
MIT
