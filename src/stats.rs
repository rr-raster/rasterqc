///Core statistics engine
use rayon::prelude::*;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StatsError {
    #[error("input raster is empty")]
    EmptyInput,
}

/// Compute min, max, mean, std, median on a raster slice.
///
/// This function is pure Rust and does not depend on Python.
pub fn compute_stats(data: &[f32]) -> Result<(f32, f32, f32, f32, f32), StatsError> {
    if data.is_empty() {
        return Err(StatsError::EmptyInput);
    }

    // --- Basic stats (min, max, sum, sum_sq) ---
    // Parallel reduction using Rayon
    let (min, max, sum, sum_sq) = data
        .par_iter()
        .fold(
            || (f32::MAX, f32::MIN, 0.0f32, 0.0f32),
            |(min, max, sum, sum_sq), &v| (min.min(v), max.max(v), sum + v, sum_sq + v * v),
        )
        .reduce(
            || (f32::MAX, f32::MIN, 0.0f32, 0.0f32),
            |a, b| (a.0.min(b.0), a.1.max(b.1), a.2 + b.2, a.3 + b.3),
        );

    let n = data.len() as f32;
    let mean = sum / n;
    let std = (sum_sq / n - mean * mean).sqrt();

    // --- Median ---
    // For even-length arrays, interpolate between the two middle values
    let mut vec = data.to_vec();
    let len = vec.len();
    use std::cmp::Ordering;
    
    vec.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    
    let median = if len % 2 == 1 {
        // Odd length: use middle element
        vec[len / 2]
    } else {
        // Even length: average the two middle elements
        let mid = len / 2;
        (vec[mid - 1] + vec[mid]) / 2.0
    };

    Ok((min, max, mean, std, median))
}

/// Optional: histogram computation
pub fn compute_histogram(data: &[f32], bins: usize) -> Result<Vec<u64>, StatsError> {
    if data.is_empty() {
        return Err(StatsError::EmptyInput);
    }

    let min = data.iter().cloned().fold(f32::MAX, f32::min);
    let max = data.iter().cloned().fold(f32::MIN, f32::max);

    let range = max - min;
    // If all values are identical, put them all in the last bin.
    if range == 0.0 {
        let mut hist = vec![0u64; bins];
        if bins > 0 {
            hist[bins - 1] = data.len() as u64;
        }
        return Ok(hist);
    }

    // Precompute inverse of bin width to replace division with a multiply.
    let inv_bin = (bins as f32) / range; // maps (v-min)*inv_bin -> [0,bins)

    let hist = data
        .par_iter()
        .fold(
            || vec![0u64; bins],
            |mut local_hist, &v| {
                // compute tentative bin index, clamp into [0, bins-1]
                let idx = ((v - min) * inv_bin) as isize;
                let idx = if idx < 0 { 0 } else { idx as usize };
                let idx = if idx >= bins { bins - 1 } else { idx };
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
