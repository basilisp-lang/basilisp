use pyo3::PyObject;
use std::hash::{Hash, Hasher};

pub struct PyTypeReference {
    wrapped: PyObject,
}

impl PyTypeReference {
    pub(crate) fn new(py_object: PyObject) -> Self {
        PyTypeReference { wrapped: py_object }
    }
}

impl Hash for PyTypeReference {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.wrapped.as_ptr().hash(state)
    }
}

impl PartialEq for PyTypeReference {
    fn eq(&self, other: &Self) -> bool {
        self.wrapped.is(&other.wrapped)
    }

    fn ne(&self, other: &Self) -> bool {
        !self.wrapped.is(&other.wrapped)
    }
}

impl Eq for PyTypeReference {}
