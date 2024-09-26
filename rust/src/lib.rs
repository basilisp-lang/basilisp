use pyo3::prelude::*;

mod singledispatch;
mod utils;

#[pymodule]
fn basilisp_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(singledispatch::singledispatch, m)?)?;
    Ok(())
}
