//PyO3 module entry point

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::exceptions::PyValueError;

use numpy::{PyReadonlyArray1, PyReadonlyArrayDyn};
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
#[pyfunction(name = "stats")]
fn stats_wrapper(py: Python<'_>, array: PyReadonlyArrayDyn<'_, f32>) -> PyResult<PyObject> {
    // Obtain an ndarray view of the array (any dimensionality)
    let view = array.as_array();

    // Require C-contiguous memory so we can borrow a slice
    let slice = view
        .as_slice_memory_order()
        .ok_or_else(|| PyValueError::new_err("array must be C-contiguous"))?;

    let (minv, maxv, mean, std, median) =
        stats::compute_stats(slice).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("min", minv)?;
    dict.set_item("max", maxv)?;
    dict.set_item("mean", mean)?;
    dict.set_item("std", std)?;
    dict.set_item("median", median)?;

    Ok(dict.into())
}

/// Optional: histogram wrapper
#[pyfunction(name = "histogram")]
fn histogram_wrapper<'py>(py: Python<'py>, array: PyReadonlyArray1<'py, f32>, bins: usize)
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
fn quantile<'py>(
    py: Python<'py>,
    array: PyReadonlyArray1<'py, f32>,
    q: f32,
) -> PyResult<f32> {
    let slice = array.as_slice()?;

    let result = py.allow_threads(|| distribution::quantile(slice, q));
    result.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn percentile<'py>(
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
fn percentiles<'py>(
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
fn iqr<'py>(
    py: Python<'py>,
    array: PyReadonlyArray1<'py, f32>,
) -> PyResult<f32> {
    let slice = array.as_slice()?;

    let result = py.allow_threads(|| distribution::iqr(slice));
    result.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Skewness
#[pyfunction]
fn skewness<'py>(
    py: Python<'py>,
    array: PyReadonlyArray1<'py, f32>,
) -> PyResult<f32> {
    let slice = array.as_slice()?;

    let result = py.allow_threads(|| distribution::skewness(slice));
    result.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}


/// Kurtosis
#[pyfunction]
fn kurtosis<'py>(
    py: Python<'py>,
    array: PyReadonlyArray1<'py, f32>,
) -> PyResult<f32> {
    let slice = array.as_slice()?;

    let result = py.allow_threads(|| distribution::kurtosis(slice));
    result.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Entropy (requires histogram counts)
#[pyfunction]
fn entropy(counts: Vec<u64>) -> PyResult<f32> {
    let result = distribution::entropy(&counts);
    result.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}


/// Python Module definition
#[pymodule]
fn rasterqc(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(stats_wrapper, m)?)?;
    m.add_function(wrap_pyfunction!(histogram_wrapper, m)?)?;
    m.add_function(wrap_pyfunction!(percentile, m)?)?;
    m.add_function(wrap_pyfunction!(quantile, m)?)?;
    m.add_function(wrap_pyfunction!(percentiles, m)?)?;
    m.add_function(wrap_pyfunction!(iqr, m)?)?;
    m.add_function(wrap_pyfunction!(skewness, m)?)?;
    m.add_function(wrap_pyfunction!(kurtosis, m)?)?;
    m.add_function(wrap_pyfunction!(entropy, m)?)?;

    // If you expose classes later, register them here

    Ok(())
}