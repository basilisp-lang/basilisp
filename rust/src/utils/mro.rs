use crate::utils::builtins::Builtins;
use crate::utils::typeref::PyTypeReference;
use crate::utils::typing::TypingModule;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::{Bound, PyObject, PyResult, Python};
use std::collections::hash_map::Keys;
use std::collections::HashSet;

pub(crate) fn get_obj_mro(cls: &Bound<'_, PyAny>) -> PyResult<HashSet<PyTypeReference>> {
    let mro: HashSet<_> = cls
        .getattr("__mro__")?
        .downcast::<PyTuple>()?
        .iter()
        .map(|item| PyTypeReference::new(item.unbind()))
        .collect();
    Ok(mro)
}

pub(crate) fn compose_mro(
    py: Python,
    cls: Bound<'_, PyAny>,
    types: Keys<PyTypeReference, PyObject>,
) -> PyResult<Vec<PyTypeReference>> {
    let builtins = Builtins::cached(py);
    let typing = TypingModule::cached(py);

    let bases: HashSet<_> = get_obj_mro(&cls)?;
    let registered_types: HashSet<_> = types.collect();
    let eligible_types: HashSet<_> = registered_types
        .iter()
        .filter(|&tref| {
            // Remove entries which are already present in the __mro__ or unrelated.
            let typ = tref.wrapped().bind(py);
            !bases.contains(tref)
                && typ.hasattr("__mro__").unwrap()
                && !typ
                    .is_instance(typing.generic_alias_type.wrapped().bind(py))
                    .unwrap()
                && builtins.issubclass(py, &cls, typ).unwrap()
        })
        .filter(|&tref| {
            // Remove entries which are strict bases of other entries (they will end up
            // in the MRO anyway).
            !registered_types.iter().any(|&other| {
                let other_mro = get_obj_mro(other.wrapped().bind(py)).unwrap();
                *tref != other && other_mro.contains(tref)
            })
        })
        .collect();
    let mro: Vec<_> = eligible_types.iter().map(|&tref| tref.clone_ref(py)).collect();
    Ok(mro)
}
