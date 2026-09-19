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

/// Compute a percentile using linear interpolation.
/// Uses the standard "nearest rank" method with linear interpolation.
pub fn percentile(data: &[f32], p: f32) -> Result<f32, DistributionError> {
    if data.is_empty() {
        return Err(DistributionError::EmptyInput);
    }
    if !(0.0..=100.0).contains(&p) {
        return Err(DistributionError::InvalidPercentile);
    }

    let mut v: Vec<f32> = data.to_vec();
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // target index with interpolation
    let rank = (p / 100.0) * ((v.len() - 1) as f32);
    let idx_low = rank.floor() as usize;
    let idx_high = rank.ceil() as usize;
    let fraction = rank - rank.floor();

    if idx_high >= v.len() {
        Ok(v[v.len() - 1])
    } else {
        Ok(v[idx_low] + fraction * (v[idx_high] - v[idx_low]))
    }
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

    // Combine second and third central moment computation into a single
    // parallel pass to reduce memory passes and iterator overhead.
    let (m2_sum, m3_sum) = data
        .par_iter()
        .fold(
            || (0.0f32, 0.0f32),
            |(s2, s3), &x| {
                let d = x - mean;
                (s2 + d * d, s3 + d * d * d)
            },
        )
        .reduce(|| (0.0f32, 0.0f32), |a, b| (a.0 + b.0, a.1 + b.1));

    let m2: f32 = m2_sum / n;
    let m3: f32 = m3_sum / n;

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

    // Compute both second and fourth central moments in a single parallel pass.
    let (m2_sum, m4_sum) = data
        .par_iter()
        .fold(
            || (0.0f32, 0.0f32),
            |(s2, s4), &x| {
                let d = x - mean;
                (s2 + d * d, s4 + d.powi(4))
            },
        )
        .reduce(|| (0.0f32, 0.0f32), |a, b| (a.0 + b.0, a.1 + b.1));

    let m2: f32 = m2_sum / n;
    let m4: f32 = m4_sum / n;

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
