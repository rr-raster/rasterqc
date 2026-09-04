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
            |(min, max, sum, sum_sq), &v| {
                (
                    min.min(v),
                    max.max(v),
                    sum + v,
                    sum_sq + v * v,
                )
            },
        )
        .reduce(
            || (f32::MAX, f32::MIN, 0.0f32, 0.0f32),
            |a, b| {
                (
                    a.0.min(b.0),
                    a.1.max(b.1),
                    a.2 + b.2,
                    a.3 + b.3,
                )
            },
        );

    let n = data.len() as f32;
    let mean = sum / n;
    let std = (sum_sq / n - mean * mean).sqrt();

    // --- Median ---
    // Fast nth-element selection instead of full sort
    //let mut vec = data.to_vec();
    //let mid = vec.len() / 2;
    //let median = {
    //    let (left, _, _) = vec.select_nth_unstable(mid);
    //    left[mid]
    //};

    // Sorted approach for median calculation. This is a simple implementation
    //  and can be replaced with a more efficient algorithm (commented out above) if needed.
    let mut sorted = data.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2]; 


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
    let bin_width = range / bins as f32;

   // let mut hist = vec![0u64; bins];

    let hist = data
    .par_iter()
    .fold(
        || vec![0u64; bins],
        |mut local_hist, &v| {
            let idx = (((v - min) / bin_width).floor() as usize).min(bins - 1);
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

