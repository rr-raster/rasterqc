// Distribution statistics module

use rayon::prelude::*;
use thiserror::Error;

/// Errors for distribution statistics
#[derive(Debug, Error)]
pub enum DistributionError {
    #[error("input slice is empty")]
    EmptyInput,

    #[error("percentile must be in [0, 100]")]
    InvalidPercentile,

    #[error("quantile must be in [0.0, 1.0]")]
    InvalidQuantile,

    #[error("failed to compute statistic")]
    ComputeError,
}

/// Compute a percentile using a sorted copy of the data.
/// This is the simplest correct implementation; you can replace it
/// with a histogram-based version later.
pub fn percentile(data: &[f32], p: f32) -> Result<f32, DistributionError> {
    if data.is_empty() {
        return Err(DistributionError::EmptyInput);
    }
    if !(0.0..=100.0).contains(&p) {
        return Err(DistributionError::InvalidPercentile);
    }

    // Copy + sort (parallel)
    let mut v: Vec<f32> = data.to_vec();
    v.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let rank = (p / 100.0) * ((v.len() - 1) as f32);
    let idx = rank.floor() as usize;

    Ok(v[idx])
}

/// Compute a quantile in [0, 1]
pub fn quantile(data: &[f32], q: f32) -> Result<f32, DistributionError> {
    if !(0.0..=1.0).contains(&q) {
        return Err(DistributionError::InvalidQuantile);
    }
    percentile(data, q * 100.0)
}

/// Interquartile range (IQR = P75 - P25)
pub fn iqr(data: &[f32]) -> Result<f32, DistributionError> {
    let p25 = percentile(data, 25.0)?;
    let p75 = percentile(data, 75.0)?;
    Ok(p75 - p25)
}

/// Skewness (Fisher-Pearson)
pub fn skewness(data: &[f32]) -> Result<f32, DistributionError> {
    if data.is_empty() {
        return Err(DistributionError::EmptyInput);
    }

    let n = data.len() as f32;
    let mean = data.par_iter().copied().sum::<f32>() / n;

    let mut m2 = 0.0f32;
    let mut m3 = 0.0f32;

    data.par_iter().for_each(|&x| {
        let d = x - mean;
        m2 += d * d;
        m3 += d * d * d;
    });

    let m2 = m2 / n;
    let m3 = m3 / n;

    if m2 == 0.0 {
        return Err(DistributionError::ComputeError);
    }

    Ok(m3 / m2.powf(1.5))
}

/// Kurtosis (excess kurtosis)
pub fn kurtosis(data: &[f32]) -> Result<f32, DistributionError> {
    if data.is_empty() {
        return Err(DistributionError::EmptyInput);
    }

    let n = data.len() as f32;
    let mean = data.par_iter().copied().sum::<f32>() / n;

    let mut m2 = 0.0f32;
    let mut m4 = 0.0f32;

    data.par_iter().for_each(|&x| {
        let d = x - mean;
        m2 += d * d;
        m4 += d.powi(4);
    });

    let m2 = m2 / n;
    let m4 = m4 / n;

    if m2 == 0.0 {
        return Err(DistributionError::ComputeError);
    }

    Ok(m4 / (m2 * m2) - 3.0)
}

/// Shannon entropy (requires histogram)
pub fn entropy(counts: &[u64]) -> Result<f32, DistributionError> {
    let total: f32 = counts.iter().map(|&c| c as f32).sum();
    if total == 0.0 {
        return Err(DistributionError::EmptyInput);
    }

    let mut ent = 0.0f32;
    for &c in counts {
        if c > 0 {
            let p = (c as f32) / total;
            ent -= p * p.ln();
        }
    }

    Ok(ent)
}
