/// Distribution statistics module
///
/// functions to compute various distribution statistics covering:
///
/// * percentiles
/// * quantiles
/// * interquartile range (IQR)
/// * skewness
/// * kurtosis
/// * Shannon entropy  
///
/// It is designed to work with slices of f32 values and u64 counts,
/// and it uses parallel computation for efficiency where applicable.  
/// The functions return results wrapped in a `Result` type to handle potential errors,
/// such as empty input or invalid parameters.
///
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

/// Compute a percentile (0–100) using linear interpolation between nearest ranks.
///
/// The data is sorted, and the percentile position is computed as:
/// `rank = p/100 * (n - 1)`. The result is a linear interpolation between the
/// lower and upper neighbouring values. `p = 0` returns the minimum; `p = 100`
/// returns the maximum.
///
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
///
/// The quantile `q` is converted to a percentile (`q * 100`).  
///
pub fn quantile(data: &[f32], q: f32) -> Result<f32, DistributionError> {
    if !(0.0..=1.0).contains(&q) {
        return Err(DistributionError::InvalidQuantile);
    }
    percentile(data, q * 100.0)
}

/// Interquartile range (IQR = P75 - P25)
///
pub fn iqr(data: &[f32]) -> Result<f32, DistributionError> {
    let p25 = percentile(data, 25.0)?;
    let p75 = percentile(data, 75.0)?;
    Ok(p75 - p25)
}

/// Skewness (Fisher-Pearson)
///
/// This computes the **third standardized moment**:
/// `skewness = m3 / m2^(3/2)`
///
/// where:
/// - `m2` is the second central moment (variance)
/// - `m3` is the third central moment
///
/// Both moments are computed in a single parallel pass for efficiency.
///
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
/// describing the "tailedness" of a distribution.
///
/// This computes the **fourth standardized moment**, returning
/// *excess* kurtosis:
///
/// `kurtosis = m4 / m2² - 3`
///
/// where:
/// - `m2` is the second central moment (variance)
/// - `m4` is the fourth central moment
///
/// Both moments are computed in a single parallel pass for efficiency.
///
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

/// Shannon entropy for a histogram of counts.
///
/// Computes:
/// `entropy = -Σ p * ln(p)`
///
/// where `p = count / total`.  
/// Zero-count bins are ignored.
///
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
