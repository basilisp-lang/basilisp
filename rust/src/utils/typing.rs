use crate::utils::typeref::PyTypeReference;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::PyTuple;
use pyo3::{Bound, IntoPyObjectExt, Py, PyAny, PyObject, PyResult, Python};

pub struct TypingModule {
    get_origin: PyObject,
    get_args: PyObject,
    pub generic_alias_type: PyTypeReference,
    union_types: Vec<PyTypeReference>,
}

static TYPING_MODULE: GILOnceCell<TypingModule> = GILOnceCell::new();

impl TypingModule {
    fn new(py: Python) -> Self {
        let typing_module = py.import("typing").unwrap();
        let types_module = py.import("types").unwrap();
        let mut union_types = Vec::with_capacity(2);
        union_types.extend([
            PyTypeReference::new(
                typing_module
                    .getattr("Union")
                    .unwrap()
                    .into_py_any(py)
                    .unwrap(),
            ),
            PyTypeReference::new(
                types_module
                    .getattr("UnionType")
                    .unwrap()
                    .into_py_any(py)
                    .unwrap(),
            ),
        ]);

        TypingModule {
            get_args: typing_module
                .getattr("get_args")
                .unwrap()
                .into_py_any(py)
                .unwrap(),
            get_origin: typing_module
                .getattr("get_origin")
                .unwrap()
                .into_py_any(py)
                .unwrap(),
            generic_alias_type: PyTypeReference::new(
                types_module
                    .getattr("GenericAlias")
                    .unwrap()
                    .into_py_any(py)
                    .unwrap(),
            ),
            union_types,
        }
    }

    pub fn cached(py: Python) -> &Self {
        TYPING_MODULE.get_or_init(py, || TypingModule::new(py))
    }

    pub fn get_args(&self, py: Python, cls: &Bound<'_, PyAny>) -> PyResult<Py<PyTuple>> {
        match self.get_args.call1(py, PyTuple::new(py, [cls])?) {
            Ok(maybe_args) => match maybe_args.downcast_bound::<PyTuple>(py) {
                Ok(args) => Ok(args.clone().unbind().clone_ref(py)),
                Err(_) => Err(PyTypeError::new_err("Expected tuple return value")),
            },
            Err(e) => Err(e),
        }
    }

    pub fn get_origin(&self, py: Python, cls: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.get_origin.call1(py, PyTuple::new(py, [cls])?)
    }

    pub fn is_union_type(&self, py: Python, cls: &Bound<'_, PyAny>) -> bool {
        let origin_type_reference = PyTypeReference::new(cls.into_py_any(py).unwrap());
        self.union_types.contains(&origin_type_reference)
    }
}
