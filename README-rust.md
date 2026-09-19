# rasterqc

Fast, parallelised raster statistics and QC metrics for scientific imaging.

## Features
- mean, median, std, skew, kurtosis
- percentiles
- histograms
- signal to noise
- uniformity
- dynamic range

## Installation
`cargo add rasterqc`

## Example
```use rasterqc::stats::compute_stats;
let data = vec![1.0, 2.0, 3.0];
let s = compute_stats(&data).unwrap();
println!("mean = {}", s.mean);
```

## CLI examples


#### Raw image array signal to noise calculation
`cat image.raw | rasterqc snr ` 

#### Dynamic range calculation
`cat image.raw | rasterqc dr` 

`cat image.raw | rasterqc dr --db`

#### Float32 array read from gdal piped to 'stats' cli function
`gdal_translate input.tif /vsistdout/ -b 1 -ot Float32 -of ENVI \
    | rasterqc stats --json`

## Documentation
https://docs.rs/rasterqc

## License
MIT
