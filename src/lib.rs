//PyO3 module entry point

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

//use numpy::{PyArray1, PyReadonlyArray1};
use numpy::{PyReadonlyArray1};
//use numpy::ndarray::ArrayView1;

mod stats;
mod histogram;
mod interop;
mod distribution;

/// Compute basic statistics (min, max, mean, std, median) on a raster.
///
/// Python signature:
///     stats(array: numpy.ndarray) -> (float, float, float, float, float)
/// (min, max, mean, std, median)
#[pyfunction]
fn stats_py<'py>(py: Python<'py>, array: PyReadonlyArray1<'py, f32>) -> PyResult<(f32, f32, f32, f32, f32)> {
    // Extract Rust slice without copying
    let slice = interop::array_to_slice(&array)?;

    // Release GIL during heavy computation
    let result = py.allow_threads(|| stats::compute_stats(slice));

    match result {
        Ok(stats) => Ok(stats),
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}

/// Optional: histogram wrapper
#[pyfunction]
fn histogram_py<'py>(py: Python<'py>, array: PyReadonlyArray1<'py, f32>, bins: usize)
    -> PyResult<Vec<u64>>
{
    let slice = interop::array_to_slice(&array)?;

    let result = py.allow_threads(|| stats::compute_histogram(slice, bins));

    match result {
        Ok(hist) => Ok(hist),
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}



/// Quantile
#[pyfunction]
fn quantile_py<'py>(
    py: Python<'py>,
    array: PyReadonlyArray1<'py, f32>,
    q: f32,
) -> PyResult<f32> {
    let slice = array.as_slice()?;

    let result = py.allow_threads(|| distribution::quantile(slice, q));
    result.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn percentile_py<'py>(
    py: Python<'py>,
    array: PyReadonlyArray1<'py, f32>,
    p: f32,
) -> PyResult<f32> {
    let slice = array.as_slice()?;
    let result = py.allow_threads(|| distribution::percentile(slice, p));
    result.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Multiple percentiles at once
#[pyfunction]
fn percentiles_py<'py>(
    py: Python<'py>,
    array: PyReadonlyArray1<'py, f32>,
    ps: Vec<f32>,
) -> PyResult<Vec<f32>> {
    let slice = array.as_slice()?;

    let result = py.allow_threads(|| {
        ps.iter()
            .map(|&p| distribution::percentile(slice, p))
            .collect::<Result<Vec<f32>, _>>()
    });

    result.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}


/// Interquartile range
#[pyfunction]
fn iqr_py<'py>(
    py: Python<'py>,
    array: PyReadonlyArray1<'py, f32>,
) -> PyResult<f32> {
    let slice = array.as_slice()?;

    let result = py.allow_threads(|| distribution::iqr(slice));
    result.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Skewness
#[pyfunction]
fn skewness_py<'py>(
    py: Python<'py>,
    array: PyReadonlyArray1<'py, f32>,
) -> PyResult<f32> {
    let slice = array.as_slice()?;

    let result = py.allow_threads(|| distribution::skewness(slice));
    result.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}


/// Kurtosis
#[pyfunction]
fn kurtosis_py<'py>(
    py: Python<'py>,
    array: PyReadonlyArray1<'py, f32>,
) -> PyResult<f32> {
    let slice = array.as_slice()?;

    let result = py.allow_threads(|| distribution::kurtosis(slice));
    result.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Entropy (requires histogram counts)
#[pyfunction]
fn entropy_py(counts: Vec<u64>) -> PyResult<f32> {
    let result = distribution::entropy(&counts);
    result.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}


/// Python Module definition
#[pymodule]
fn rasterqc(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(stats_py, m)?)?;
    m.add_function(wrap_pyfunction!(histogram_py, m)?)?;
    m.add_function(wrap_pyfunction!(percentile_py, m)?)?;
    m.add_function(wrap_pyfunction!(quantile_py, m)?)?;
    m.add_function(wrap_pyfunction!(percentiles_py, m)?)?;
    m.add_function(wrap_pyfunction!(iqr_py, m)?)?;
    m.add_function(wrap_pyfunction!(skewness_py, m)?)?;
    m.add_function(wrap_pyfunction!(kurtosis_py, m)?)?;
    m.add_function(wrap_pyfunction!(entropy_py, m)?)?;

    // If you expose classes later, register them here

    Ok(())
}