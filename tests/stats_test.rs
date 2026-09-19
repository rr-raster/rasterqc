use rasterqc_core::stats::{compute_stats, compute_histogram, StatsError};

#[test]
fn test_basic_stats_small_array() {
    let data = vec![1.0, 2.0, 3.0, 4.0];

    let (min, max, mean, std, median) = compute_stats(&data).unwrap();

    assert_eq!(min, 1.0);
    assert_eq!(max, 4.0);
    assert_eq!(mean, 2.5);
    assert!((std - 1.118034).abs() < 1e-6); // population std
    assert_eq!(median, 2.5);
}

#[test]
fn test_stats_single_value() {
    let data = vec![42.0];
    let (min, max, mean, std, median) = compute_stats(&data).unwrap();

    assert_eq!(min, 42.0);
    assert_eq!(max, 42.0);
    assert_eq!(mean, 42.0);
    assert_eq!(median, 42.0);
    assert_eq!(std, 0.0);
}

#[test]
fn test_stats_empty_input() {
    let data: Vec<f32> = vec![];
    let err = compute_stats(&data).unwrap_err();
    match err {
        StatsError::EmptyInput => {}
        _ => panic!("Expected EmptyInput error"),
    }
}

#[test]
fn test_stats_unsorted_input() {
    let data = vec![10.0, -1.0, 3.0, 7.0];

    let (min, max, mean, std, median) = compute_stats(&data).unwrap();

    assert_eq!(min, -1.0);
    assert_eq!(max, 10.0);
    assert_eq!(mean, 4.75);
    assert!((std - 4.147288).abs() < 1e-6);
    assert_eq!(median, 5.0); // sorted: [-1, 3, 7, 10]
}
