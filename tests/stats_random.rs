use rasterqc::stats::{compute_histogram, compute_stats};

#[test]
fn test_stats_large_random() {
    let mut data = Vec::with_capacity(100_000);
    for i in 0..100_000 {
        data.push((i as f32).sin());
    }

    let result = compute_stats(&data);
    assert!(result.is_ok());
}

#[test]
fn test_histogram_large_random() {
    let mut data = Vec::with_capacity(200_000);
    for i in 0..200_000 {
        data.push((i as f32).cos());
    }

    let hist = compute_histogram(&data, 128);
    assert!(hist.is_ok());
}
