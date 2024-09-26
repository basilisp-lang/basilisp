use pyo3::exceptions::PyTypeError;
use pyo3::prelude::{PyAnyMethods, PyTupleMethods};
use pyo3::sync::GILOnceCell;
use pyo3::types::{PyDict, PyTuple};
use pyo3::{
    pyclass, pyfunction, pymethods, Bound, IntoPy, Py, PyAny, PyObject, PyResult, Python,
    ToPyObject,
};
use std::collections::HashMap;
use std::sync::Mutex;

use crate::utils::typeref::PyTypeReference;

static PY_OBJECT_TYPE: GILOnceCell<PyObject> = GILOnceCell::new();

fn get_py_object_type(py: Python) -> &PyObject {
    PY_OBJECT_TYPE.get_or_init(py, || {
        py.import_bound("builtins")
            .unwrap()
            .getattr("object")
            .unwrap()
            .to_object(py)
    })
}

fn get_abc_cache_token(py: Python) -> Bound<'_, PyAny> {
    py.import_bound("abc")
        .unwrap()
        .getattr("get_cache_token")
        .unwrap()
        .call0()
        .unwrap()
}

struct SingleDispatchState {
    registry: HashMap<PyTypeReference, PyObject>,
    cache: HashMap<PyTypeReference, PyObject>,
    cache_token: Option<PyObject>,
}

#[pyclass]
pub(crate) struct SingleDispatch {
    lock: Mutex<SingleDispatchState>,
}

impl SingleDispatch {
    fn register_cls(
        &self,
        py: Python<'_>,
        cls: Bound<'_, PyAny>,
        func: Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        match self.lock.lock() {
            Ok(mut state) => {
                let unbound_func = func.unbind();
                state.registry.insert(
                    PyTypeReference::new(cls.into_py(py)),
                    unbound_func.clone_ref(py),
                );
                Ok(unbound_func)
            }
            Err(_) => panic!("Singledispatch mutex poisoned!"),
        }
    }
}

#[pymethods]
impl SingleDispatch {
    #[new]
    fn __new__<'py>(py: Python, func: Bound<'py, PyAny>) -> Self {
        let mut registry = HashMap::new();
        let py_object_type = get_py_object_type(py).clone_ref(py);
        let f = func.unbind();
        registry.insert(PyTypeReference::new(py_object_type), f);

        SingleDispatch {
            lock: Mutex::new(SingleDispatchState {
                registry,
                cache: HashMap::new(),
                cache_token: None,
            }),
        }
    }

    #[pyo3(signature = (obj, /, *args, **kwargs))]
    fn __call__(
        &self,
        py: Python<'_>,
        obj: Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        match obj.getattr("__class__") {
            Ok(cls) => {
                let mut all_args = Vec::with_capacity(1 + args.len());
                all_args.insert(0, obj);
                all_args.extend(args);

                match self.dispatch(py, cls) {
                    Ok(handler) => handler.call_bound(py, PyTuple::new_bound(py, all_args), kwargs),
                    Err(_) => panic!("no handler for singledispatch"),
                }
            }
            Err(_) => Err(PyTypeError::new_err("expected __class__ attribute for obj")),
        }
    }

    fn dispatch(&self, py: Python<'_>, cls: Bound<'_, PyAny>) -> PyResult<PyObject> {
        match self.lock.lock() {
            Ok(state) => match state.registry.get(&PyTypeReference::new(cls.unbind())) {
                Some(handler) => Ok(handler.clone_ref(py)),
                None => {
                    let py_object_type = get_py_object_type(py);
                    let default_handler = state
                        .registry
                        .get(&PyTypeReference::new(py_object_type.clone_ref(py)));
                    assert!(default_handler.is_some());
                    Ok(default_handler.unwrap().clone_ref(py))
                }
            },
            Err(_) => panic!("Singledispatch mutex poisoned!"),
        }
    }

    #[pyo3(signature = (cls, func=None))]
    fn register(
        slf: Py<Self>,
        py: Python<'_>,
        cls: Bound<'_, PyAny>,
        func: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let singledispatch = slf.borrow(py);
        match func {
            Some(actual_func) => singledispatch.register_cls(py, cls, actual_func),
            None => {
                Ok(PartialSingleDispatchRegistration::__new__(slf.clone_ref(py), cls).into_py(py))
            }
        }
    }
}

#[pyclass]
struct PartialSingleDispatchRegistration {
    singledispatch: Py<SingleDispatch>,
    cls: PyObject,
}

#[pymethods]
impl PartialSingleDispatchRegistration {
    #[new]
    fn __new__<'py>(singledispatch: Py<SingleDispatch>, cls: Bound<'py, PyAny>) -> Self {
        PartialSingleDispatchRegistration {
            singledispatch,
            cls: cls.unbind(),
        }
    }

    #[pyo3(signature = (func))]
    fn __call__(&self, py: Python<'_>, func: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let singledispatch = self.singledispatch.borrow(py);
        singledispatch.register_cls(py, self.cls.clone_ref(py).into_bound(py), func)
    }
}

#[pyfunction]
pub(crate) fn singledispatch<'py>(py: Python, func: Bound<'py, PyAny>) -> PyResult<SingleDispatch> {
    Ok(SingleDispatch::__new__(py, func))
}
