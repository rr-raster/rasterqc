use rasterqc::distribution::{percentile, DistributionError};

#[test]
fn test_percentile_basic() {
    let data = vec![1.0, 2.0, 3.0, 4.0];

    assert_eq!(percentile(&data, 0.0).unwrap(), 1.0);
    assert_eq!(percentile(&data, 100.0).unwrap(), 4.0);
    assert_eq!(percentile(&data, 50.0).unwrap(), 2.5);
}

#[test]
fn test_percentile_unsorted() {
    let data = vec![10.0, 0.0, 5.0, 20.0];
    let p50 = percentile(&data, 50.0).unwrap();
    assert_eq!(p50, 7.5); // sorted: [0,5,10,20]
}

#[test]
fn test_percentile_invalid_range() {
    let data = vec![1.0, 2.0, 3.0];

    assert!(matches!(
        percentile(&data, -1.0),
        Err(DistributionError::InvalidPercentile)
    ));
    assert!(matches!(
        percentile(&data, 101.0),
        Err(DistributionError::InvalidPercentile)
    ));
}

#[test]
fn test_percentile_empty() {
    let data: Vec<f32> = vec![];
    assert!(matches!(
        percentile(&data, 50.0),
        Err(DistributionError::EmptyInput)
    ));
}
