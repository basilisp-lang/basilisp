use crate::utils::mro::{compose_mro, get_obj_mro};
use crate::utils::typeref::PyTypeReference;
use crate::utils::typing::TypingModule;
use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyTypeError};
use pyo3::prelude::*;

use crate::utils::builtins::Builtins;
use pyo3::types::{PyDict, PyTuple, PyType};
use pyo3::{
    pyclass, pyfunction, pymethods, Bound, IntoPyObjectExt, Py, PyAny, PyObject, PyResult, Python,
};
use std::collections::HashMap;
use std::sync::Mutex;

fn get_abc_cache_token(py: Python) -> Bound<'_, PyAny> {
    py.import("abc")
        .unwrap()
        .getattr("get_cache_token")
        .unwrap()
        .call0()
        .unwrap()
}

fn valid_dispatch_types(py: Python, cls: &Bound<'_, PyAny>) -> PyResult<Vec<Py<PyType>>> {
    if let Ok(typ) = cls.downcast::<PyType>() {
        Ok(Vec::from([typ.clone().unbind()]))
    } else {
        let typing_module = TypingModule::cached(py);
        if let Ok(origin) = typing_module.get_origin(py, cls) {
            if typing_module.is_union_type(py, origin.bind(py)) {
                if let Ok(type_args) = typing_module.get_args(py, cls) {
                    let py_tuple = type_args.bind(py);
                    let mut dispatch_types = Vec::with_capacity(py_tuple.len());
                    for (i, item) in py_tuple.iter().enumerate() {
                        match item.downcast::<PyType>() {
                            Ok(typ) => {
                                dispatch_types.insert(i, typ.clone().unbind());
                            }
                            Err(_) => {
                                return Err(PyTypeError::new_err(format!(
                                    "Object {item} is not a valid type"
                                )))
                            }
                        }
                    }
                    Ok(dispatch_types)
                } else {
                    Ok(Vec::new())
                }
            } else {
                Ok(Vec::new())
            }
        } else {
            Ok(Vec::new())
        }
    }
}

fn is_valid_dispatch_type(py: Python, cls: &Bound<'_, PyAny>) -> bool {
    if let Ok(types) = valid_dispatch_types(py, cls) {
        !types.is_empty()
    } else {
        false
    }
}

struct SingleDispatchState {
    registry: HashMap<PyTypeReference, PyObject>,
    cache: HashMap<PyTypeReference, PyObject>,
    cache_token: Option<PyObject>,
}

impl SingleDispatchState {
    fn find_impl(&mut self, py: Python, cls: Bound<'_, PyAny>) -> PyResult<PyObject> {
        let cls_mro = get_obj_mro(&cls.clone()).unwrap();
        let mro = compose_mro(py, cls.clone(), self.registry.keys())?;
        let mut mro_match: Option<PyTypeReference> = None;
        for typ in mro.iter() {
            if self.registry.contains_key(typ) {
                mro_match = Some(typ.clone_ref(py));
            }

            if mro_match.is_some() {
                let m = &mro_match.unwrap().clone_ref(py);
                if self.registry.contains_key(typ)
                    && !cls_mro.contains(typ)
                    && !cls_mro.contains(m)
                    && Builtins::cached(py)
                        .issubclass(py, m.wrapped().bind(py), typ.wrapped().bind(py))
                        .is_ok_and(|res| res)
                {
                    return Err(PyRuntimeError::new_err(format!(
                        "Ambiguous dispatch: {m} or {typ}"
                    )));
                }
                mro_match = Some(m.clone_ref(py));
                break;
            }
        }
        match mro_match {
            Some(_) => match self.registry.get(&mro_match.unwrap()) {
                Some(&ref it) => Ok(it.clone_ref(py)),
                None => Ok(py.None()),
            },
            None => Ok(py.None()),
        }
    }

    fn get_or_find_impl(&mut self, py: Python, cls: Bound<'_, PyAny>) -> PyResult<PyObject> {
        let free_cls = cls.unbind();
        let type_reference = PyTypeReference::new(free_cls.clone_ref(py));

        match self.cache.get(&type_reference) {
            Some(handler) => Ok(handler.clone_ref(py)),
            None => {
                let handler_for_cls = match self.registry.get(&type_reference) {
                    Some(handler) => handler.clone_ref(py),
                    None => self.find_impl(py, free_cls.bind(py).clone())?,
                };
                self.cache
                    .insert(type_reference, handler_for_cls.clone_ref(py));
                Ok(handler_for_cls)
            }
        }
    }
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
        let typing_module = TypingModule::cached(py);
        match self.lock.lock() {
            Ok(mut state) => {
                let unbound_func = func.unbind();
                if typing_module.is_union_type(py, &cls) {
                    match typing_module.get_args(py, &cls) {
                        Ok(tuple) => {
                            for tp in tuple.bind(py).iter() {
                                state.registry.insert(
                                    PyTypeReference::new(tp.unbind()),
                                    unbound_func.clone_ref(py),
                                );
                            }
                        }
                        Err(e) => return Err(e),
                    }
                } else {
                    state.registry.insert(
                        PyTypeReference::new(cls.into_pyobject(py)?.unbind()),
                        unbound_func.clone_ref(py),
                    );
                }
                if state.cache_token.is_none() {
                    if let Ok(_) = unbound_func.getattr(py, "__abstractmethods__") {
                        state.cache_token = Some(get_abc_cache_token(py).unbind());
                    }
                }
                state.cache.clear();
                Ok(unbound_func)
            }
            Err(_) => panic!("Singledispatch mutex poisoned!"),
        }
    }

    fn register_with_type_annotations(
        &self,
        _py: Python<'_>,
        cls: Bound<'_, PyAny>,
        func: Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        match func.getattr("__annotations__") {
            Ok(_annotations) => Err(PyNotImplementedError::new_err("Oops!")),
            Err(_) => Err(PyTypeError::new_err(
                format!("Invalid first argument to `register()`: {cls}. Use either `@register(some_class)` or plain `@register` on an annotated function."),
            )),
        }
    }
}

#[pymethods]
impl SingleDispatch {
    #[new]
    fn __new__<'py>(py: Python, func: Bound<'py, PyAny>) -> Self {
        let mut registry = HashMap::new();
        let py_object_type = Builtins::cached(py).object_type.clone_ref(py);
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
                    Ok(handler) => handler.call(py, PyTuple::new(py, all_args)?, kwargs),
                    Err(_) => panic!("no handler for singledispatch"),
                }
            }
            Err(_) => Err(PyTypeError::new_err("expected __class__ attribute for obj")),
        }
    }

    fn dispatch(&self, py: Python<'_>, cls: Bound<'_, PyAny>) -> PyResult<PyObject> {
        match self.lock.lock() {
            Ok(mut state) => {
                match &state.cache_token {
                    Some(cache_token) => {
                        let current_token = get_abc_cache_token(py);
                        match current_token.rich_compare(cache_token.bind(py), CompareOp::Eq) {
                            Ok(_) => {
                                state.cache.clear();
                                state.cache_token = Some(current_token.unbind());
                            }
                            _ => (),
                        }
                    }
                    _ => (),
                }

                state.get_or_find_impl(py, cls)
            }
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
        if is_valid_dispatch_type(py, &cls) {
            match func {
                Some(actual_func) => singledispatch.register_cls(py, cls, actual_func),
                None => match PartialSingleDispatchRegistration::__new__(slf.clone_ref(py), cls)
                    .into_pyobject(py)
                {
                    Ok(v) => Ok(v.into_py_any(py)?),
                    Err(_) => Err(PyRuntimeError::new_err("")),
                },
            }
        } else {
            match func {
                Some(f) => singledispatch.register_with_type_annotations(py, cls, f),
                None => Err(PyTypeError::new_err(format!(
                    "invalid first argument to `register()`. {cls} must be a class or union type."
                ))),
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
