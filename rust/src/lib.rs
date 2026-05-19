use pyo3::prelude::*;

mod basilisp_native;

#[pymodule]
mod _lang {
    #[pymodule_export]
    use super::seq;
}

#[pymodule]
mod seq {
    #[pymodule_export]
    pub use crate::basilisp_native::seq::{to_seq, LazySeq};
}
