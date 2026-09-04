use numpy::PyReadonlyArray1;
use thiserror::Error;
use pyo3::exceptions::PyValueError;
use pyo3::PyErr;

#[derive(Debug, Error)]
pub enum InteropError {
    #[error("array is not contiguous")]
    NotContiguous,

    #[error("failed to extract slice from numpy array")]
    SliceError,
}

impl From<InteropError> for PyErr {
    fn from(err: InteropError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

pub fn array_to_slice<'py>(array: &'py PyReadonlyArray1<'py, f32>) -> Result<&'py [f32], InteropError> {
    if !array.is_c_contiguous() {
        return Err(InteropError::NotContiguous);
    }

    array.as_slice().map_err(|_| InteropError::SliceError)
}
