//PyO3 module entry point

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

//use numpy::{PyArray1, PyReadonlyArray1};
use numpy::{PyReadonlyArray1};
//use numpy::ndarray::ArrayView1;

mod stats;
mod histogram;
mod interop;

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

/// Python module definition
#[pymodule]
fn rasterstats(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(stats_py, m)?)?;
    m.add_function(wrap_pyfunction!(histogram_py, m)?)?;

    // If you expose classes later, register them here

    Ok(())
}
