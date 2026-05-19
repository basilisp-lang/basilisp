use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyType};
use pyo3::{intern, PyTypeInfo};
use std::fmt::Debug;
use std::mem;
use std::sync::Mutex;

use super::interfaces::{is_iseq, is_iseqable};


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

#[derive(Debug)]
struct LazySeqState {
    gen: Option<Py<PyAny>>,
    obj: Option<Py<PyAny>>,
    seq: Option<Py<PyAny>>,
}

impl LazySeqState {
    fn _compute_seq(&mut self, py: Python) -> PyResult<Py<PyAny>> {
        // println!("_compute_seq: self = {:?}", self);
        if let Some(_) = self.gen {
            let gen = self.gen.as_ref().unwrap().clone_ref(py);
            self.gen = None;
            self.obj = Some(gen.call0(py)?);
            // println!("_compute_seq: self = {:?}", self);
        }
        // println!("_compute_seq: self = {:?}", self);
        if let Some(o) = &self.obj {
            Ok(o.clone_ref(py))
        } else {
            Ok(self.seq.as_ref().unwrap().clone_ref(py))
        }
    }

    fn seq(&mut self, py: Python) -> PyResult<Py<PyAny>> {
        let _ = self._compute_seq(py)?;
        // println!("computed = {}", computed);
        if let Some(_) = self.obj {
            let mut wrapped: Option<Py<PyAny>> = None;
            mem::swap(&mut self.obj, &mut wrapped);
            let mut o = wrapped.unwrap();
            let lazy_seq_tp = LAZY_SEQ_TYPE
                .get_or_init(py, || LazySeq::type_object(py).unbind())
                .bind(py);
            loop {
                // println!("obj = {}", o.getattr(py, "__class__")?);
                if o.bind(py).is_instance(lazy_seq_tp)? {
                    // println!("is_lazy_seq = true");
                    o = o.call_method0(py, intern!(py, "_compute_seq"))?;
                } else {
                    // println!("is_lazy_seq = false");
                    break;
                }
            }
            // println!("LazySeqState.seq() = {}", o);

            self.seq = Some(to_seq(py, o.bind(py))?.unbind());
            // println!("self.seq = {}", self.seq.as_ref().unwrap());
        }
        Ok(self.seq.as_ref().unwrap().clone_ref(py))
    }

    fn is_empty(&mut self, py: Python) -> PyResult<bool> {
        Ok(self.seq(py)?.is_none(py))
    }

    fn first(&mut self, py: Python) -> PyResult<Py<PyAny>> {
        match self.seq(py) {
            Ok(v) => if v.is_none(py) {
                Ok(py.None())
            } else {
                v.getattr(py, intern!(py, "first"))
            }
            Err(e) => Err(e),
        }
    }

    fn rest(&mut self, py: Python) -> PyResult<Py<PyAny>> {
        match self.seq(py) {
            Ok(v) => if v.is_none(py) {
                Ok(empty_seq(py).clone().unbind())
            } else {
                v.getattr(py, intern!(py, "rest"))
            }
            Err(e) => Err(e),
        }
    }

    fn is_realized(&self) -> bool {
        self.gen.is_none()
    }
}

static EMPTY_SEQ: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

#[pyclass(subclass, module = "basilisp._basilisp_native.seq")]
pub struct LazySeq {
    lock: Mutex<LazySeqState>,
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
            // println!("gen = {:?}; seq = {:?}", gen, seq);
            Ok(LazySeq {
                lock: Mutex::new(LazySeqState {
                    gen: if gen.is_none() {
                        None
                    } else {
                        Some(gen.unbind())
                    },
                    obj: None,
                    seq: if seq.is_none() {
                        None
                    } else {
                        Some(seq.unbind())
                    },
                }),
                meta: meta.unbind(),
            })
        }
    }

    fn _compute_seq(&self, py: Python) -> PyResult<Py<PyAny>> {
        match self.lock.lock() {
            Ok(mut state) => state._compute_seq(py),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "LazySeq mutex poisoned: {e}"
            ))),
        }
    }

    fn seq(&self, py: Python) -> PyResult<Py<PyAny>> {
        match self.lock.lock() {
            Ok(mut state) => state.seq(py),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "LazySeq mutex poisoned: {e}"
            ))),
        }
    }

    #[getter(meta)]
    fn meta<'py>(&self, py: Python<'py>) -> PyResult<&Bound<'py, PyAny>> {
        Ok(self.meta.bind(py))
    }

    fn with_meta(&self, py: Python, meta: Bound<'_, PyAny>) -> PyResult<Self> {
        match self.lock.lock() {
            Ok(_) => Ok(LazySeq {
                lock: Mutex::new(LazySeqState {
                    gen: None,
                    obj: None,
                    seq: Some(self.seq(py)?),
                }),
                meta: meta.unbind(),
            }),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "LazySeq mutex poisoned: {e}"
            ))),
        }
    }

    #[getter(first)]
    fn first(&mut self, py: Python) -> PyResult<Py<PyAny>> {
        match self.lock.lock() {
            Ok(mut state) => state.first(py),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "LazySeq mutex poisoned: {e}"
            ))),
        }
    }

    #[getter(rest)]
    fn rest(&mut self, py: Python) -> PyResult<Py<PyAny>> {
        match self.lock.lock() {
            Ok(mut state) => state.rest(py),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "LazySeq mutex poisoned: {e}"
            ))),
        }
    }

    #[getter(is_empty)]
    fn is_empty<'py>(&mut self, py: Python<'py>) -> PyResult<Borrowed<'py, 'py, PyBool>> {
        match self.lock.lock() {
            Ok(mut state) => Ok(PyBool::new(py, state.is_empty(py)?)),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "LazySeq mutex poisoned: {e}"
            ))),
        }
    }

    #[getter(is_realized)]
    fn is_realized<'py>(&mut self, py: Python<'py>) -> PyResult<Borrowed<'py, 'py, PyBool>> {
        match self.lock.lock() {
            Ok(state) => Ok(PyBool::new(py, state.is_realized())),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "LazySeq mutex poisoned: {e}"
            ))),
        }
    }

    fn empty<'py>(&self, py: Python<'py>) -> &Bound<'py, PyAny> {
        empty_seq(py)
    }
}

static LAZY_SEQ_TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();

#[pyfunction]
fn seq_or_nil(py: Python, s: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // println!("seq_or_nil: s = {}", s);
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
    // println!("=> to_seq({})", s);
    if s.is_none() {
        // println!("<= None");
        Ok(py.None().into_bound(py))
    } else if s.is_instance(
        LAZY_SEQ_TYPE
            .get_or_init(py, || LazySeq::type_object(py).unbind())
            .bind(py),
    )? {
        // println!("<= LazySeq.seq()");
        s.call_method0(intern!(py, "seq"))
    } else if is_iseq(py, s)? {
        // println!("<= already ISeq");
        Ok(seq_or_nil(py, &s)?.bind(py).clone())
    } else if is_iseqable(py, s)? {
        let seq = s.call_method0(intern!(py, "seq"))?.clone();
        // println!("<= ISeqable.seq() = {}", seq);
        Ok(seq_or_nil(py, &seq)?.bind(py).clone())
    } else {
        // println!("<= default");
        Ok(seq_or_nil(py, s)?.bind(py).clone())
    }
}
