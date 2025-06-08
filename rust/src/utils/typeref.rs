use pyo3::{PyObject, Python};
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};

pub struct PyTypeReference {
    wrapped: PyObject,
}

impl PyTypeReference {
    pub(crate) fn new(py_object: PyObject) -> Self {
        PyTypeReference { wrapped: py_object }
    }

    pub(crate) fn wrapped(&self) -> &PyObject {
        &self.wrapped
    }

    pub(crate) fn clone_ref(&self, py: Python) -> Self {
        Self {
            wrapped: self.wrapped.clone_ref(py),
        }
    }
}

impl Display for PyTypeReference {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.wrapped, f)
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
