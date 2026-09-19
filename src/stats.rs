/// Core statistics module
///
/// Provides fast, parallel computation of basic descriptive statistics for
/// raster and imaging data. The functions operate on slices of `f32` values. 
/// Functions implemented:
///
/// * compute_stats: min, max, mean, std, median
/// * compute_histogram: histogram counts for a given number of bins
///
/// The implementation uses Rayon to parallelize reductions, minimizing
/// memory passes and improving throughput on large datasets. Median
/// computation is performed after sorting a local copy of the data.
///
use rayon::prelude::*;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StatsError {
    #[error("input raster is empty")]
    EmptyInput,
}

/// Compute basic descriptive statistics:
///  `min`, `max`, `mean`, `std`, and `median`.
///
/// This function performs parallel reduction to compute:
/// - **min** and **max**  
/// - **mean** (average value)  
/// - **standard deviation** (population σ)  
///
/// Median is computed after sorting a local copy of the data:
/// - odd-length slices return the middle element  
/// - even-length slices return the average of the two middle values  
///
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

/// Compute a fixed‑width histogram over a slice of `f32` values.
///
/// The histogram is divided into `bins` equally spaced intervals between the
/// minimum and maximum values in the data. The function performs:
///
/// * parallel reduction to find `min` and `max`  
/// * precomputation of the inverse bin width for fast bin indexing  
/// * per‑thread local histograms merged in a final reduction  
///
/// Special cases:
/// * returns an error if the input is empty  
/// * if all values are identical, all counts are placed in the last bin  
///
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
