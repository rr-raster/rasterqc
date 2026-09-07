// Rust Histogram computation module
// 
use rayon::prelude::*;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HistogramError {
    #[error("input raster is empty")]
    EmptyInput,

    #[error("number of bins must be > 0")]
    InvalidBins,
}

/// Compute a histogram over a raster slice.
/// It uses parallel iteration for speed.
///
/// Arguments:
///     data: &[f32]      - raster values
///     bins: usize       - number of histogram bins
///
/// Returns:
///     Vec<u64>          - histogram counts
#[allow(dead_code)]
pub fn compute_histogram(data: &[f32], bins: usize) -> Result<Vec<u64>, HistogramError> {
    if data.is_empty() {
        return Err(HistogramError::EmptyInput);
    }
    if bins == 0 {
        return Err(HistogramError::InvalidBins);
    }

    let (min, max) = data
        .par_iter()
        .cloned()
        .fold(|| (f32::MAX, f32::MIN), |(mn, mx), v| (mn.min(v), mx.max(v)))
        .reduce(|| (f32::MAX, f32::MIN), |a, b| (a.0.min(b.0), a.1.max(b.1)));

    let range = max - min;
    let bin_width = range / bins as f32;

    let hist = data
        .par_iter()
        .fold(
            || vec![0u64; bins],
            |mut local_hist, &v| {
                let idx = if bin_width > 0.0 {
                    (((v - min) / bin_width).floor() as usize).min(bins - 1)
                } else {
                    0
                };
                local_hist[idx] += 1;
                local_hist
            },
        )
        .reduce(
            || vec![0u64; bins],
            |mut a, b| {
                for i in 0..bins {
                    a[i] += b[i];
                }
                a
            },
        );

    Ok(hist)
}

