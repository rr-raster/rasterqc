/// PyO3 bindings for raster quality control functions.
/// This module provides a Python interface to the Rust-based raster quality 
// control functions implemented in the `rasterqc` crate. 
// It allows users to compute various statistics, histograms, 
// and quality metrics on raster data efficiently using Rust's performance and safety features.


use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::exceptions::PyValueError;
use numpy::{PyReadonlyArrayDyn, PyArrayDyn};

mod stats;
mod histogram;
mod distribution;
mod quality;

/// Zero-copy slice extraction.
/// NumPy arrays are already contiguous and row-major.
/// `as_slice()` gives a fully flattened &[f32].
fn get_slice<'a>(array: &'a PyReadonlyArrayDyn<'a, f32>) -> Result<&'a [f32], PyErr> {
    array
        .as_slice()
        .map_err(|_| PyValueError::new_err("array must be C-contiguous"))
}

/// Zero-copy slice extraction for u64 arrays (used for histogram counts / entropy)
fn get_u64_slice<'a>(array: &'a PyReadonlyArrayDyn<'a, u64>) -> Result<&'a [u64], PyErr> {
    array
        .as_slice()
        .map_err(|_| PyValueError::new_err("array must be C-contiguous"))
}

#[pyfunction(name = "stats")]
fn stats_wrapper(py: Python<'_>, array: PyReadonlyArrayDyn<'_, f32>) -> PyResult<PyObject> {
    let slice = get_slice(&array)?;

    let (minv, maxv, mean, std, median) =
        stats::compute_stats(slice).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("min", minv)?;
    dict.set_item("max", maxv)?;
    dict.set_item("mean", mean)?;
    dict.set_item("std", std)?;
    dict.set_item("median", median)?;

    Ok(dict.to_object(py))
}

#[pyfunction(name = "stats_fast")]
fn stats_fast(_py: Python<'_>, array: PyReadonlyArrayDyn<'_, f32>)
    -> PyResult<(f32, f32, f32, f32, f32)>
{
    let slice = get_slice(&array)?;
    stats::compute_stats(slice)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction(name = "histogram")]
fn histogram_wrapper(py: Python<'_>, array: PyReadonlyArrayDyn<'_, f32>, bins: usize)
    -> PyResult<Vec<u64>>
{
    let slice = get_slice(&array)?;
    py.allow_threads(|| histogram::compute_histogram(slice, bins))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn percentile(py: Python<'_>, array: PyReadonlyArrayDyn<'_, f32>, p: f32) -> PyResult<f32> {
    let slice = get_slice(&array)?;
    py.allow_threads(|| distribution::percentile(slice, p))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn quantile(py: Python<'_>, array: PyReadonlyArrayDyn<'_, f32>, q: f32) -> PyResult<f32> {
    let slice = get_slice(&array)?;
    py.allow_threads(|| distribution::quantile(slice, q))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn skewness(py: Python<'_>, array: PyReadonlyArrayDyn<'_, f32>) -> PyResult<f32> {
    let slice = get_slice(&array)?;
    py.allow_threads(|| distribution::skewness(slice))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn kurtosis(py: Python<'_>, array: PyReadonlyArrayDyn<'_, f32>) -> PyResult<f32> {
    let slice = get_slice(&array)?;
    py.allow_threads(|| distribution::kurtosis(slice))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn iqr(py: Python<'_>, array: PyReadonlyArrayDyn<'_, f32>) -> PyResult<f32> {
    let slice = get_slice(&array)?;
    py.allow_threads(|| distribution::iqr(slice))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn entropy(py: Python<'_>, counts: &PyAny) -> PyResult<f32> {
    // Accept either a NumPy uint64 array or any sequence (list/tuple) of integers.
    // First try to downcast to a NumPy array for zero-copy access.
    if let Ok(pyarr) = counts.downcast::<PyArrayDyn<u64>>() {
        let readonly = pyarr.readonly();
        let slice = readonly.as_slice().map_err(|_| PyValueError::new_err("array must be C-contiguous"))?;
        return py.allow_threads(|| distribution::entropy(slice))
            .map_err(|e| PyValueError::new_err(e.to_string()));
    }

    // Fallback: try to extract a Vec<u64> from any sequence (this will allocate).
    let vec: Vec<u64> = counts.extract()
        .map_err(|_| PyValueError::new_err("counts must be a NumPy uint64 array or an iterable of integers"))?;

    py.allow_threads(|| distribution::entropy(&vec))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn snr(py: Python<'_>, array: PyReadonlyArrayDyn<'_, f32>) -> PyResult<f32> {
    let slice = get_slice(&array)?;
    py.allow_threads(|| quality::snr(slice))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn dynamic_range(py: Python<'_>, array: PyReadonlyArrayDyn<'_, f32>) -> PyResult<f32> {
    let slice = get_slice(&array)?;
    py.allow_threads(|| quality::dynamic_range(slice))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn dynamic_range_db(py: Python<'_>, array: PyReadonlyArrayDyn<'_, f32>) -> PyResult<f32> {
    let slice = get_slice(&array)?;
    py.allow_threads(|| quality::dynamic_range_db(slice))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn uniformity(py: Python<'_>, array: PyReadonlyArrayDyn<'_, f32>) -> PyResult<f32> {
    let slice = get_slice(&array)?;
    py.allow_threads(|| quality::uniformity(slice))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}


#[pymodule]
fn rasterqc(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(stats_wrapper, m)?)?;
    m.add_function(wrap_pyfunction!(stats_fast, m)?)?;
    m.add_function(wrap_pyfunction!(histogram_wrapper, m)?)?;
    m.add_function(wrap_pyfunction!(percentile, m)?)?;
    m.add_function(wrap_pyfunction!(quantile, m)?)?;
    m.add_function(wrap_pyfunction!(skewness, m)?)?;
    m.add_function(wrap_pyfunction!(kurtosis, m)?)?;
    m.add_function(wrap_pyfunction!(iqr, m)?)?;
    m.add_function(wrap_pyfunction!(entropy, m)?)?;
    m.add_function(wrap_pyfunction!(snr, m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_range, m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_range_db, m)?)?;
    m.add_function(wrap_pyfunction!(uniformity, m)?)?;
    Ok(())
}
