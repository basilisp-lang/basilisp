use crate::utils::typeref::PyTypeReference;
use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyTypeError};
use pyo3::prelude::{PyAnyMethods, PyTupleMethods};
use pyo3::sync::GILOnceCell;
use pyo3::types::{PyDict, PyTuple, PyType};
use pyo3::{
    pyclass, pyfunction, pymethods, Bound, IntoPy, Py, PyAny, PyObject, PyResult, Python,
    ToPyObject,
};
use std::collections::HashMap;
use std::sync::Mutex;

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

struct TypingModule {
    get_origin: PyObject,
    get_args: PyObject,
    union_types: Vec<PyTypeReference>,
}

static TYPING_MODULE: GILOnceCell<TypingModule> = GILOnceCell::new();

impl TypingModule {
    fn new(py: Python) -> Self {
        let typing_module = py.import_bound("typing").unwrap();
        let types_module = py.import_bound("types").unwrap();
        let mut union_types = Vec::with_capacity(2);
        union_types.extend([
            PyTypeReference::new(typing_module.getattr("Union").unwrap().to_object(py)),
            PyTypeReference::new(types_module.getattr("UnionType").unwrap().to_object(py)),
        ]);

        TypingModule {
            get_args: typing_module.getattr("get_args").unwrap().to_object(py),
            get_origin: typing_module.getattr("get_origin").unwrap().to_object(py),
            union_types,
        }
    }

    fn cached(py: Python) -> &Self {
        TYPING_MODULE.get_or_init(py, || TypingModule::new(py))
    }

    fn get_args(&self, py: Python, cls: &Bound<'_, PyAny>) -> PyResult<Py<PyTuple>> {
        match self.get_args.call1(py, PyTuple::new_bound(py, [cls])) {
            Ok(maybe_args) => match maybe_args.downcast_bound::<PyTuple>(py) {
                Ok(args) => Ok(args.clone().unbind().clone_ref(py)),
                Err(_) => Err(PyTypeError::new_err("Expected tuple return value")),
            },
            Err(e) => Err(e),
        }
    }

    fn get_origin(&self, py: Python, cls: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.get_origin.call1(py, PyTuple::new_bound(py, [cls]))
    }

    fn is_union_type(&self, py: Python, cls: &Bound<'_, PyAny>) -> bool {
        let origin_type_reference = PyTypeReference::new(cls.into_py(py));
        self.union_types.contains(&origin_type_reference)
    }
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
    fn find_impl(&mut self, py: Python) -> PyResult<PyObject> {
        let py_object_type = get_py_object_type(py);
        let default_handler = self
            .registry
            .get(&PyTypeReference::new(py_object_type.clone_ref(py)));
        assert!(default_handler.is_some());
        Ok(default_handler.unwrap().clone_ref(py))
    }

    fn get_or_find_impl(&mut self, py: Python, cls: Bound<'_, PyAny>) -> PyResult<PyObject> {
        let type_reference = PyTypeReference::new(cls.unbind());

        match self.cache.get(&type_reference) {
            Some(handler) => Ok(handler.clone_ref(py)),
            None => {
                let handler_for_cls = match self.registry.get(&type_reference) {
                    Some(handler) => handler.clone_ref(py),
                    None => match self.find_impl(py) {
                        Ok(handler) => handler,
                        Err(_) => return Err(PyRuntimeError::new_err("Something bad happened!")),
                    },
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
                        PyTypeReference::new(cls.into_py(py)),
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
                None => Ok(
                    PartialSingleDispatchRegistration::__new__(slf.clone_ref(py), cls).into_py(py),
                ),
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
