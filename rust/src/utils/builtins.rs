use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::{PyBool, PyTuple};
use pyo3::{IntoPyObjectExt, PyObject, Python};

pub struct Builtins {
    pub object_type: PyObject,
    issubclass_func: PyObject,
}

static PY_BUILTINS: GILOnceCell<Builtins> = GILOnceCell::new();

impl Builtins {
    fn new(py: Python) -> Self {
        let builtins_module = py.import("builtins").unwrap();
        Builtins {
            object_type: builtins_module
                .getattr("object")
                .unwrap()
                .into_py_any(py)
                .unwrap(),
            issubclass_func: builtins_module
                .getattr("issubclass")
                .unwrap()
                .into_py_any(py)
                .unwrap(),
        }
    }

    pub fn cached(py: Python) -> &Self {
        PY_BUILTINS.get_or_init(py, || Builtins::new(py))
    }

    pub fn issubclass(
        &self,
        py: Python,
        cls: &Bound<'_, PyAny>,
        typ: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let args = PyTuple::new(py, [cls, typ]);
        match self.issubclass_func.call1(py, args?) {
            Ok(result) => Ok(result.downcast_bound::<PyBool>(py).unwrap().is_true()),
            Err(e) => Err(e),
        }
    }
}
