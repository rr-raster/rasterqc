/// Quality metrics module
///
/// Functions for computing image and raster quality measures, including:
///
/// * signal‑to‑noise ratio (SNR)
/// * uniformity metrics
/// * dynamic range
/// * contrast and intensity measures
///
/// These functions operate on slices of `f32` values and are designed for
/// scientific imaging workflows where fast, parallel computation is useful.
/// All results are returned in a `Result` to handle invalid inputs such as
/// empty data or degenerate values.

use rayon::prelude::*;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StatsError {
    #[error("input raster is empty")]
    EmptyInput,
    #[error("unable to compute statistic")]
    ComputeError,
}

/// Signal‑to‑Noise Ratio (SNR), 
/// defined as `mean / standard_deviation (SNR = mu / sigma)
///
/// This computes:
/// - the mean intensity
/// - the variance and standard deviation
///
/// SNR is a common quality metric in imaging and raster analysis, expressing
/// how strong the signal is relative to noise. 
pub fn snr(data: &[f32]) -> Result<f32, StatsError> {
    if data.is_empty() {
        return Err(StatsError::EmptyInput);
    }

    let n = data.len() as f32;
    // Mean
    let mean = data.par_iter().copied().sum::<f32>() / n;
    // Variance
    let var = data
        .par_iter()
        .map(|&x| {
            let d = x - mean;
            d * d
        })
        .sum::<f32>()
        / n;
    // Standard deviation
    let std = var.sqrt();

    if std == 0.0 {
        return Err(StatsError::ComputeError);
    }

    Ok(mean / std)
}


/// Dynamic range, computed as `max / min`.
///
/// This measures the ratio between the brightest and darkest values in the
/// data, a common quality indicator in imaging and raster analysis.
///
/// The minimum and maximum are computed in a single parallel pass
///
pub fn dynamic_range(data: &[f32]) -> Result<f32, StatsError> {
    if data.is_empty() {
        return Err(StatsError::EmptyInput);
    }

    let (minv, maxv) = data
        .par_iter()
        .copied()
        .fold(
            || (f32::MAX, f32::MIN),
            |(min, max), v| (min.min(v), max.max(v)),
        )
        .reduce(|| (f32::MAX, f32::MIN), |a, b| (a.0.min(b.0), a.1.max(b.1)));

    if minv <= 0.0 {
        return Err(StatsError::ComputeError);
    }

    Ok(maxv / minv)
}

/// Dynamic range expressed in decibels (dB).
///
/// This converts the linear dynamic range `max / min` into decibels using:
/// `20 * log10(dynamic_range)`.
///
/// Useful in imaging and signal analysis where ratios are commonly reported
/// on a logarithmic scale.
///
pub fn dynamic_range_db(data: &[f32]) -> Result<f32, StatsError> {
    let dr = dynamic_range(data)?;
    Ok(20.0 * dr.log10())
}

/// Uniformity metric, defined as `1 - (σ / μ)`.
///
/// This measures how consistent the values are relative to their mean:
/// - `μ` is the mean intensity  
/// - `σ` is the standard deviation  
///
/// The result lies in `[0, 1]`, where:
/// - `1` indicates perfect uniformity (all values identical)  
/// - lower values indicate greater variation  
///
pub fn uniformity(data: &[f32]) -> Result<f32, StatsError> {
    if data.is_empty() {
        return Err(StatsError::EmptyInput);
    }

    let n = data.len() as f32;

    let mean = data.par_iter().copied().sum::<f32>() / n;

    let var = data
        .par_iter()
        .map(|&x| {
            let d = x - mean;
            d * d
        })
        .sum::<f32>()
        / n;

    let std = var.sqrt();

    if mean == 0.0 {
        return Err(StatsError::ComputeError);
    }

    Ok(1.0 - (std / mean))
}
