use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::sync::{PyOnceLock, RwLockExt};
use pyo3::types::{PyBool, PyType};
use pyo3::{intern, PyTypeInfo};
use std::ops::{Deref, DerefMut};
use std::sync::RwLock;

use super::interfaces::{is_iseq, is_iseqable};

static EMPTY_SEQ: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static LAZY_SEQ_TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static SEQUENCE_FN: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

fn empty_seq<'py>(py: Python<'py>) -> &'py Bound<'py, PyAny> {
    EMPTY_SEQ
        .get_or_init(py, || {
            py.import("basilisp.lang.seq")
                .unwrap()
                .getattr("EMPTY")
                .unwrap()
                .unbind()
        })
        .bind(py)
}

#[pyfunction]
fn seq_or_nil(py: Python, s: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if s.is_none() {
        Ok(s.clone().unbind())
    } else if s
        .getattr(intern!(s.py(), "is_empty"))?
        .cast::<PyBool>()?
        .is_true()
    {
        Ok(py.None())
    } else {
        Ok(s.clone().unbind())
    }
}

#[pyfunction]
pub fn to_seq<'py>(py: Python<'py>, s: &'py Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if s.is_none() {
        Ok(py.None().into_bound(py))
    } else if s.is_instance(
        LAZY_SEQ_TYPE
            .get_or_init(py, || LazySeq::type_object(py).unbind())
            .bind(py),
    )? {
        s.call_method0(intern!(py, "seq"))
    } else if is_iseq(py, s)? {
        Ok(seq_or_nil(py, &s)?.bind(py).clone())
    } else if is_iseqable(py, s)? {
        let seq = s.call_method0(intern!(py, "seq"))?.clone();
        Ok(seq_or_nil(py, &seq)?.bind(py).clone())
    } else {
        let sequence_fn = SEQUENCE_FN.get_or_init(py, || {
            py.import("basilisp.lang.seq")
                .unwrap()
                .getattr(intern!(py, "sequence"))
                .unwrap()
                .unbind()
        });
        Ok(seq_or_nil(py, &sequence_fn.bind(py).call1((s,))?)?
            .bind(py)
            .clone())
    }
}

enum LazySeqState {
    Initialized(Py<PyAny>),
    Computed(Py<PyAny>),
    Realized(Py<PyAny>),
}

#[pyclass(subclass, module = "basilisp._lang.seq")]
pub struct LazySeq {
    lock: RwLock<LazySeqState>,
    meta: Py<PyAny>,
}

#[pymethods]
impl LazySeq {
    #[new]
    #[pyo3(signature = (gen, seq, *, meta))]
    fn __new__<'py>(
        gen: Bound<'py, PyAny>,
        seq: Bound<'py, PyAny>,
        meta: Bound<'py, PyAny>,
    ) -> PyResult<Self> {
        if !gen.is_none() && !seq.is_none() {
            Err(PyTypeError::new_err(
                "cannot construct LazySeq with generator function and realized seq",
            ))
        } else {
            Ok(LazySeq {
                lock: RwLock::new(if gen.is_none() {
                    LazySeqState::Realized(seq.unbind())
                } else {
                    LazySeqState::Initialized(gen.unbind())
                }),
                meta: meta.unbind(),
            })
        }
    }

    fn _compute_seq(&self, py: Python) -> PyResult<Py<PyAny>> {
        for _ in 0..10 {
            {
                match self.lock.try_read() {
                    Ok(v) => match v.deref() {
                        LazySeqState::Initialized(_) => (),
                        LazySeqState::Computed(obj) => {
                            return Ok(obj.clone_ref(py));
                        }
                        LazySeqState::Realized(seq) => {
                            return Ok(seq.as_ref().clone_ref(py));
                        }
                    },
                    Err(std::sync::TryLockError::Poisoned(inner)) => {
                        return Err(PyRuntimeError::new_err(format!(
                            "LazySeq mutex poisoned: {inner}"
                        )))
                    }
                    Err(_) => (),
                }
            }

            {
                match self.lock.try_write() {
                    Ok(mut state) => {
                        if let LazySeqState::Initialized(gen) = state.deref_mut() {
                            let obj = gen.call0(py)?;
                            *state = LazySeqState::Computed(obj.clone_ref(py));
                            return Ok(obj.clone_ref(py));
                        }
                    }
                    Err(std::sync::TryLockError::Poisoned(inner)) => {
                        return Err(PyRuntimeError::new_err(format!(
                            "LazySeq mutex poisoned: {inner}"
                        )))
                    }
                    Err(_) => (),
                }
            }
        }

        Err(PyRuntimeError::new_err(format!("Unable to acquire lock!")))
        // panic!("Unable to acquire lock!")
    }

    fn seq(&self, py: Python) -> PyResult<Py<PyAny>> {
        for _ in 0..10 {
            {
                match self.lock.try_read() {
                    Ok(v) => {
                        if let LazySeqState::Realized(seq) = v.deref() {
                            return Ok(seq.as_ref().clone_ref(py));
                        }
                    }
                    Err(std::sync::TryLockError::Poisoned(inner)) => {
                        return Err(PyRuntimeError::new_err(format!(
                            "LazySeq mutex poisoned: {inner}"
                        )))
                    }
                    Err(_) => (),
                }
            }

            self._compute_seq(py)?;

            {
                match self.lock.try_write() {
                    Ok(mut state) => {
                        if let LazySeqState::Computed(ref mut obj) = state.deref_mut() {
                            let lazy_seq_tp = LAZY_SEQ_TYPE
                                .get_or_init(py, || LazySeq::type_object(py).unbind())
                                .bind(py);
                            loop {
                                if obj.bind(py).is_instance(lazy_seq_tp)? {
                                    *obj = obj.call_method0(py, intern!(py, "_compute_seq"))?;
                                } else {
                                    break;
                                }
                            }
                            let result = to_seq(py, obj.bind(py))?.unbind();
                            *state = LazySeqState::Realized(result.clone_ref(py));
                            return Ok(result.clone_ref(py));
                        }
                    }
                    Err(std::sync::TryLockError::Poisoned(inner)) => {
                        return Err(PyRuntimeError::new_err(format!(
                            "LazySeq mutex poisoned: {inner}"
                        )))
                    }
                    Err(_) => (),
                }
            }
        }

        Err(PyRuntimeError::new_err(format!("Unable to acquire lock!")))
        // panic!("Unable to acquire lock!")
    }

    #[getter(meta)]
    fn meta<'py>(&self, py: Python<'py>) -> PyResult<&Bound<'py, PyAny>> {
        Ok(self.meta.bind(py))
    }

    #[getter(first)]
    fn first(&self, py: Python) -> PyResult<Py<PyAny>> {
        match self.seq(py) {
            Ok(v) => {
                if v.is_none(py) {
                    Ok(py.None())
                } else {
                    v.getattr(py, intern!(py, "first"))
                }
            }
            Err(e) => Err(e),
        }
    }

    #[getter(rest)]
    fn rest(&self, py: Python) -> PyResult<Py<PyAny>> {
        match self.seq(py) {
            Ok(v) => {
                if v.is_none(py) {
                    Ok(empty_seq(py).clone().unbind())
                } else {
                    v.getattr(py, intern!(py, "rest"))
                }
            }
            Err(e) => Err(e),
        }
    }

    #[getter(is_empty)]
    fn is_empty<'py>(&self, py: Python<'py>) -> PyResult<Borrowed<'py, 'py, PyBool>> {
        Ok(PyBool::new(py, self.seq(py)?.is_none(py)))
    }

    #[getter(is_realized)]
    fn is_realized<'py>(&mut self, py: Python<'py>) -> PyResult<Borrowed<'py, 'py, PyBool>> {
        match self.lock.read_py_attached(py) {
            Ok(v) => Ok(PyBool::new(
                py,
                if let LazySeqState::Realized(_) = v.deref() {
                    true
                } else {
                    false
                },
            )),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "LazySeq mutex poisoned: {e}"
            ))),
        }
    }

    fn empty<'py>(&self, py: Python<'py>) -> &Bound<'py, PyAny> {
        empty_seq(py)
    }
}
