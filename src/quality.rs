use rayon::prelude::*;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StatsError {
    #[error("input raster is empty")]
    EmptyInput,
    #[error("unable to compute statistic")]
    ComputeError,
}

/// Compute the Signal-to-Noise Ratio (SNR)
/// SNR = mu / sigma
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

/// Compute dynamic range in decibels (dB)
pub fn dynamic_range_db(data: &[f32]) -> Result<f32, StatsError> {
    let dr = dynamic_range(data)?;
    Ok(20.0 * dr.log10())
}

/// Compute uniformity of the data
/// Uniformity = 1 - (𝜎 / q𝜇)
/// where 𝜎 is the standard deviation, 𝜇 is the mean, and q is a quality factor (default q=1)
/// Returns a value between 0 and 1, where 1 indicates perfect uniformity.
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
