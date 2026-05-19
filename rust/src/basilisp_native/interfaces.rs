use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyType;
use pyo3::{intern, PyResult};

static INTERFACES_MODULE: PyOnceLock<Py<PyModule>> = PyOnceLock::new();

fn interfaces_module(py: Python<'_>) -> &Py<PyModule> {
    INTERFACES_MODULE.get_or_init(py, || {
        py.import(intern!(py, "basilisp.lang.interfaces"))
            .unwrap()
            .unbind()
    })
}

static ISEQ_TYPE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

pub fn is_iseq(py: Python, s: &'_ Bound<'_, PyAny>) -> PyResult<bool> {
    s.is_instance(
        ISEQ_TYPE
            .get_or_init(py, || {
                interfaces_module(py)
                    .getattr(py, intern!(py, "ISeq"))
                    .unwrap()
            })
            .cast_bound::<PyType>(py)?,
    )
}

static ISEQABLE_TYPE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

pub fn is_iseqable(py: Python, s: &'_ Bound<'_, PyAny>) -> PyResult<bool> {
    s.is_instance(
        ISEQABLE_TYPE
            .get_or_init(py, || {
                interfaces_module(py)
                    .getattr(py, intern!(py, "ISeqable"))
                    .unwrap()
            })
            .cast_bound::<PyType>(py)?,
    )
}
