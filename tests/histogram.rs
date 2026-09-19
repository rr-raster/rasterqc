use rasterqc::histogram::{compute_histogram, HistogramError};

#[test]
fn test_histogram_uniform() {
    let data = vec![0.0, 0.5, 1.0];
    let hist = compute_histogram(&data, 2).unwrap();

    assert_eq!(hist.len(), 2);
    // range [0, 0.5) → bin 0, range [0.5, 1.0] → bin 1
    assert_eq!(hist[0], 1); // 0.0
    assert_eq!(hist[1], 2); // 0.5 and 1.0
}

#[test]
fn test_histogram_all_same_bin() {
    let data = vec![0.1, 0.1, 0.1];
    let hist = compute_histogram(&data, 10).unwrap();

    // All values fall into the same bin
    let sum: u64 = hist.iter().sum();
    assert_eq!(sum, 3);
}

#[test]
fn test_histogram_empty() {
    let data: Vec<f32> = vec![];
    assert!(matches!(compute_histogram(&data, 10), Err(HistogramError::EmptyInput)));
}

#[test]
fn test_histogram_edge_values() {
    let data = vec![0.0, 1.0];
    let hist = compute_histogram(&data, 2).unwrap();

    assert_eq!(hist[0], 1); // 0.0
    assert_eq!(hist[1], 1); // 1.0
}
