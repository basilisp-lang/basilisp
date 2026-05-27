use pyo3::prelude::*;

mod basilisp_native;

#[pymodule]
mod _lang {
    use pyo3::prelude::*;

    #[pymodule]
    mod seq {
        #[pymodule_export]
        pub use crate::basilisp_native::seq::{
            sequence, to_seq, Cons, EmptySequence, LazySeq, SeqIterator,
        };
    }

    /// Allow importing basilisp._lang.seq directly.
    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        let modules = PyModule::import(m.py(), "sys")?.getattr("modules")?;
        modules.set_item("basilisp._lang.seq", m.getattr("seq")?)?;
        Ok(())
    }
}
