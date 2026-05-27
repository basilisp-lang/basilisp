use super::interfaces::{is_iseq, is_iseqable};
use parking_lot::ReentrantMutex;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyDict, PyIterator, PyTuple, PyType};
use pyo3::{intern, IntoPyObjectExt, PyTypeInfo};
use std::cell::RefCell;
use std::ops::Deref;

static CONS_TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static EMPTY_SEQ: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static LAZY_SEQ_TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static PY_LAZY_SEQ_TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();

/// Create a new Cons type from the final class type defined in Python.
///
/// Rust types cannot inherit from Python types as of PyO3 0.28.3, so in order to
/// have the ISeq types defined in Rust correctly implement the ISeq interface (and
/// other interfaces), we define the "final" class type in Python. This creates a
/// circular reference where the Rust types need to have references to the final
/// type in order to create new versions of the same from Rust code.
///
/// This function is a bit of a kludge to enable that construction.
fn new_py_cons<'py>(
    py: Python<'py>,
    first: Bound<'py, PyAny>,
    rest: Option<Bound<'py, PyAny>>,
    meta: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let tp = CONS_TYPE
        .get_or_init(py, || {
            py.import("basilisp.lang.seq")
                .unwrap()
                .getattr("Cons")
                .unwrap()
                .cast()
                .unwrap()
                .clone()
                .unbind()
        })
        .bind(py);

    match meta {
        Some(m) => {
            let d = PyDict::new(py);
            d.set_item("meta", m)?;
            tp.call((first, rest), Some(&d))
        }
        None => tp.call((first, rest), None),
    }
}

/// Create a new LazySeq type from the final class type defined in Python.
///
/// See the docstring for `new_py_cons` for more details as to why this is necessary.
fn new_py_lazy_seq<'py>(py: Python<'py>, gen: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let tp = PY_LAZY_SEQ_TYPE
        .get_or_init(py, || {
            py.import("basilisp.lang.seq")
                .unwrap()
                .getattr("LazySeq")
                .unwrap()
                .cast()
                .unwrap()
                .clone()
                .unbind()
        })
        .bind(py);

    tp.call1((gen,))
}

/// Return a statically defined empty seq from Python.
///
/// See the docstring for `new_py_cons` for more details as to why this is necessary.
fn empty_seq<'py>(py: Python<'py>) -> &'py Bound<'py, PyAny> {
    EMPTY_SEQ
        .get_or_init(py, || {
            py.import("basilisp.lang.seq")
                .unwrap()
                .getattr("EMPTY")
                .unwrap()
                .unbind()
        })
        .bind(py)
}

/// Given a seq, return that seq if it contains elements or Python None otherwise.
fn seq_or_nil<'py>(py: Python<'py>, s: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if s.is_none() {
        Ok(s.clone())
    } else if s
        .getattr(intern!(s.py(), "is_empty"))?
        .cast::<PyBool>()?
        .is_true()
    {
        Ok(py.None().into_bound(py))
    } else {
        Ok(s.clone())
    }
}

/// Coerce a value to a seq or return None otherwise.
#[pyfunction]
pub fn to_seq<'py>(py: Python<'py>, s: &'py Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if s.is_none() {
        Ok(py.None().into_bound(py))
    } else if s.is_instance(
        LAZY_SEQ_TYPE
            .get_or_init(py, || LazySeq::type_object(py).unbind())
            .bind(py),
    )? {
        s.call_method0(intern!(py, "seq"))
    } else if is_iseq(py, s)? {
        Ok(seq_or_nil(py, s)?)
    } else if is_iseqable(py, s)? {
        let seq = s.call_method0(intern!(py, "seq"))?.clone();
        Ok(seq_or_nil(py, &seq)?)
    } else {
        Ok(seq_or_nil(py, &sequence(py, s.clone(), None)?)?)
    }
}

#[pyclass(subclass, module = "basilisp._lang.seq")]
pub struct Sequence {
    it: Py<PyIterator>,
}

#[pymethods]
impl Sequence {
    #[new]
    #[pyo3(signature = (s, support_single_use=None))]
    fn __new__<'py>(s: Bound<'py, PyAny>, support_single_use: Option<bool>) -> PyResult<Self> {
        let it = s.try_iter()?;

        if !support_single_use.unwrap_or(false) && it.is(s.clone()) {
            return Err(PyTypeError::new_err(format!(
                "Can't create sequence out of single-use iterable object, please use iterator-seq instead. Iterable Object type: {}",
                s.get_type()
            )));
        }

        Ok(Sequence { it: it.unbind() })
    }

    fn __call__<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mut it = slf.it.bind(py).clone();
        match it.next() {
            Some(Ok(v)) => Ok(new_py_cons(
                py,
                v,
                Some(new_py_lazy_seq(py, slf.into_bound_py_any(py)?)?),
                None,
            )?),
            Some(Err(e)) => Err(e),
            None => Ok(empty_seq(py).clone()),
        }
    }
}

/// Create a seq from Iterable `s`, wrapping the Iterable in successive LazySeqs.
///
/// By default, raise a TypeError if `s` is a single-use Iterable, unless
/// `support_single_use` is ``True``.
#[pyfunction]
#[pyo3(signature = (s, support_single_use=None))]
pub fn sequence<'py>(
    py: Python<'py>,
    s: Bound<'py, PyAny>,
    support_single_use: Option<bool>,
) -> PyResult<Bound<'py, PyAny>> {
    let seq = Sequence::__new__(s, support_single_use)?;
    new_py_lazy_seq(py, seq.into_bound_py_any(py)?)
}

/// Stateful iterator for sequence types.
///
/// This is primarily useful for avoiding blowing the stack on a long (or infinite)
/// sequence. It is not safe to use `yield` statements to iterate over sequences,
/// since they accrete one Python stack frame per sequence element.
#[pyclass(subclass, generic, module = "basilisp._lang.seq")]
pub struct SeqIterator {
    cur: Py<PyAny>,
}

#[pymethods]
impl SeqIterator {
    #[new]
    #[pyo3(signature = (cur))]
    fn __new__<'py>(cur: Bound<'py, PyAny>) -> Self {
        SeqIterator { cur: cur.unbind() }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python) -> PyResult<Option<Py<PyAny>>> {
        if slf.cur.is_none(py) {
            return Ok(None);
        }

        // Call `seq` on the current element first and then check `is_empty`/`first`/`rest`
        // afterward. This is primarily important for LazySeqs, where each of the property
        // accesses will attempt to grab the inner lock. Grabbing `seq` and then accessing
        // those properties directly means we only take the lock once, which should
        // improve throughput on iteration over LazySeqs.
        let s = slf.cur.call_method0(py, intern!(py, "seq"))?;
        if s.is_none(py)
            || s.getattr(py, intern!(py, "is_empty"))?
                .cast_bound::<PyBool>(py)?
                .is_true()
        {
            return Ok(None);
        }

        let v = s.getattr(py, intern!(py, "first"))?;
        let r = s.getattr(py, intern!(py, "rest"))?;
        slf.cur = r;
        Ok(Some(v))
    }
}

/// An empty seq.
///
/// Generally referenced using the static value `basilisp.lang.seq.EMPTY` rather
/// than created dynamically.
#[pyclass(subclass, generic, frozen, module = "basilisp._lang.seq")]
pub struct EmptySequence {
    meta: Py<PyAny>,
}

#[pymethods]
impl EmptySequence {
    #[new]
    #[pyo3(signature = (*, meta=None))]
    fn __new__<'py>(py: Python, meta: Option<Bound<'py, PyAny>>) -> Self {
        EmptySequence {
            meta: match meta {
                Some(v) => v.unbind(),
                None => py.None(),
            },
        }
    }

    fn __iter__(slf: PyRef<'_, Self>, py: Python) -> PyResult<SeqIterator> {
        Ok(SeqIterator {
            cur: slf.into_py_any(py)?,
        })
    }

    #[getter(is_empty)]
    fn is_empty(&self) -> bool {
        true
    }

    #[getter(first)]
    fn first(&self, py: Python) -> Py<PyAny> {
        py.None()
    }

    #[getter(rest)]
    fn rest<'py>(&self, py: Python<'py>) -> &Bound<'py, PyAny> {
        empty_seq(py)
    }

    #[pyo3(signature = (*elems))]
    fn cons<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
        elems: &Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut eiter = elems.iter();
        match eiter.nth(0) {
            Some(e) => {
                let mut l = new_py_cons(py, e, Some(slf.clone().into_bound_py_any(py)?), None)?;
                for elem in elems.iter().skip(1) {
                    l = new_py_cons(py, elem, Some(l.into_bound_py_any(py)?), None)?;
                }
                Ok(l.into_bound_py_any(py)?)
            }
            None => Ok(empty_seq(py).clone()),
        }
    }

    fn empty<'py>(&self, py: Python<'py>) -> &Bound<'py, PyAny> {
        empty_seq(py)
    }

    #[getter(meta)]
    fn meta<'py>(&self, py: Python<'py>) -> PyResult<&Bound<'py, PyAny>> {
        Ok(self.meta.bind(py))
    }

    fn with_meta<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
        meta: Py<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let tp = slf.get_type();
        let kwargs = PyDict::new(py);
        kwargs.set_item("meta", meta)?;
        tp.call((), Some(&kwargs))
    }
}

#[pyclass(subclass, generic, frozen, module = "basilisp._lang.seq")]
pub struct Cons {
    first: Py<PyAny>,
    rest: Option<Py<PyAny>>,
    meta: Py<PyAny>,
}

#[pymethods]
impl Cons {
    #[new]
    #[pyo3(signature = (first, rest=None, *, meta=None))]
    fn __new__<'py>(
        py: Python,
        first: Bound<'py, PyAny>,
        rest: Option<Bound<'py, PyAny>>,
        meta: Option<Bound<'py, PyAny>>,
    ) -> Self {
        Cons {
            first: Py::from(first),
            rest: match rest {
                Some(r) => {
                    if r.is_none() {
                        None
                    } else {
                        Some(Py::from(r))
                    }
                }
                None => None,
            },
            meta: match meta {
                Some(v) => v.unbind(),
                None => py.None(),
            },
        }
    }

    fn __iter__(slf: PyRef<'_, Self>, py: Python) -> PyResult<SeqIterator> {
        Ok(SeqIterator {
            cur: slf.into_py_any(py)?,
        })
    }

    #[getter(is_empty)]
    fn is_empty(&self) -> bool {
        false
    }

    #[getter(first)]
    fn first(&self, py: Python) -> Py<PyAny> {
        self.first.clone_ref(py)
    }

    #[getter(rest)]
    fn rest(&self, py: Python) -> Py<PyAny> {
        match &self.rest {
            Some(r) => r.clone_ref(py),
            None => empty_seq(py).clone().unbind(),
        }
    }

    #[pyo3(signature = (*elems))]
    fn cons<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
        elems: &Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut eiter = elems.iter();
        match eiter.nth(0) {
            Some(e) => {
                let mut l = new_py_cons(py, e, Some(slf.clone().into_bound_py_any(py)?), None)?;
                for elem in elems.iter().skip(1) {
                    l = new_py_cons(py, elem, Some(l.into_bound_py_any(py)?), None)?;
                }
                Ok(l.into_bound_py_any(py)?)
            }
            None => Ok(empty_seq(py).clone()),
        }
    }

    fn empty<'py>(&self, py: Python<'py>) -> &Bound<'py, PyAny> {
        empty_seq(py)
    }

    #[getter(meta)]
    fn meta<'py>(&self, py: Python<'py>) -> PyResult<&Bound<'py, PyAny>> {
        Ok(self.meta.bind(py))
    }

    fn with_meta<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
        meta: Py<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let tp = slf.get_type();
        let kwargs = PyDict::new(py);
        kwargs.set_item("meta", meta)?;
        let cur = slf.borrow();
        tp.call(
            (
                cur.first.clone_ref(py),
                cur.rest.as_ref().unwrap().clone_ref(py),
            ),
            Some(&kwargs),
        )
    }
}

enum LazySeqState {
    Initialized(Py<PyAny>),
    Computing,
    Computed(Py<PyAny>),
    Realized(Py<PyAny>),
}

#[pyclass(subclass, generic, frozen, module = "basilisp._lang.seq")]
pub struct LazySeq {
    lock: ReentrantMutex<RefCell<LazySeqState>>,
    meta: Py<PyAny>,
}

#[pymethods]
impl LazySeq {
    #[new]
    #[pyo3(signature = (gen, seq=None, *, meta=None))]
    fn __new__<'py>(
        py: Python<'py>,
        gen: Bound<'py, PyAny>,
        seq: Option<Bound<'py, PyAny>>,
        meta: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Self> {
        if !gen.is_none() && seq.is_some() && seq.clone().unwrap().is_none() {
            Err(PyTypeError::new_err(
                "cannot construct LazySeq with generator function and realized seq",
            ))
        } else {
            Ok(LazySeq {
                lock: ReentrantMutex::new(RefCell::new(if gen.is_none() {
                    LazySeqState::Realized(seq.unwrap().unbind())
                } else {
                    LazySeqState::Initialized(gen.unbind())
                })),
                meta: match meta {
                    Some(v) => v.unbind(),
                    None => py.None(),
                },
            })
        }
    }

    fn __iter__(slf: PyRef<'_, Self>, py: Python) -> PyResult<SeqIterator> {
        Ok(SeqIterator {
            cur: slf.into_py_any(py)?,
        })
    }

    // LazySeqs have a fairly complex inner state, in spite of the simple interface.
    // Calls from Basilisp code should be providing the only generator seed function.
    // Calls to `(seq ...)` cause the LazySeq to cache the generator function locally
    // (as explained in _compute_seq), clear it, and cache the results of that generator
    // function call. The result may be None, some other ISeq, or perhaps another
    // LazySeq. Finally, the LazySeq attempts to consume all returned LazySeq objects
    // before calling `(seq ...)` on the result, which is cached.

    fn _compute_seq(&self, py: Python) -> PyResult<Py<PyAny>> {
        let mutex = self.lock.lock();
        let state = mutex.borrow();
        match state.deref() {
            LazySeqState::Computing => return Ok(py.None()),
            LazySeqState::Computed(obj) => {
                return Ok(obj.clone_ref(py));
            }
            LazySeqState::Realized(seq) => {
                return Ok(seq.as_ref().clone_ref(py));
            }
            _ => (),
        }
        drop(state);

        // This local caching of the generator function and clearing of the generator
        // is absolutely critical for supporting co-recursive lazy sequences.
        //
        // The original example that prompted this change is below:
        //
        //   (def primes (remove
        //                (fn [x] (some #(zero? (mod x %)) primes))
        //                (iterate inc 2)))
        //
        //   (take 5 primes)  ;; => stack overflow
        //
        // If we don't clear the generator, each successive call to (some ... primes)
        // will end up forcing the primes LazySeq object to call the generator, rather
        // than caching the results, allowing examination of the partial seq
        // computed up to that point.
        let mut state = mutex.borrow_mut();
        let mut genfn: Option<Py<PyAny>> = None;
        if let LazySeqState::Initialized(gen) = state.deref() {
            genfn = Some(gen.clone_ref(py));
            *state = LazySeqState::Computing;
        }
        drop(state);

        if let Some(gen) = genfn {
            let obj = gen.call0(py)?;
            let mut state = mutex.borrow_mut();
            *state = LazySeqState::Computed(obj.clone_ref(py));
            Ok(obj.clone_ref(py))
        } else {
            panic!("Expected a reference to a generator function!");
        }
    }

    fn seq(&self, py: Python) -> PyResult<Py<PyAny>> {
        let mutex = self.lock.lock();
        let state = mutex.borrow();
        if let LazySeqState::Realized(seq) = state.deref() {
            return Ok(seq.as_ref().clone_ref(py));
        }
        drop(state);

        self._compute_seq(py)?;

        let state = mutex.borrow();
        match state.deref() {
            LazySeqState::Computed(obj) => {
                let mut wrapped = obj.clone_ref(py);
                let lazy_seq_tp = LAZY_SEQ_TYPE
                    .get_or_init(py, || LazySeq::type_object(py).unbind())
                    .bind(py);
                drop(state); // Drop the borrow while we compute intermediate seqs.

                // Consume any additional lazy sequences returned immediately, so we
                // have a "real" concrete sequence to proxy to.
                //
                // The common idiom with LazySeqs is to return
                // (cons value (lazy-seq ...)) from the generator function, so this will
                // only result in evaluating away instances where _another_ LazySeq is
                // returned rather than a cons cell with a concrete first value. This
                // loop will not consume the LazySeq in the rest position of the cons.
                loop {
                    if wrapped.bind(py).is_instance(lazy_seq_tp)? {
                        wrapped = wrapped.call_method0(py, intern!(py, "_compute_seq"))?;
                    } else {
                        break;
                    }
                }

                // Mutably borrow the object again so we can update the inner state.
                let mut state = mutex.borrow_mut();
                let result = to_seq(py, wrapped.bind(py))?.unbind();
                *state = LazySeqState::Realized(result.clone_ref(py));
                Ok(result.clone_ref(py))
            }
            _ => Ok(py.None()),
        }
    }

    #[getter(meta)]
    fn meta<'py>(&self, py: Python<'py>) -> PyResult<&Bound<'py, PyAny>> {
        Ok(self.meta.bind(py))
    }

    fn with_meta<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
        meta: Py<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let tp = slf.get_type();
        let kwargs = PyDict::new(py);
        kwargs.set_item("meta", meta)?;
        let cur = slf.borrow();
        tp.call((py.None(), cur.seq(py)?), Some(&kwargs))
    }

    #[getter(first)]
    fn first(&self, py: Python) -> PyResult<Py<PyAny>> {
        match self.seq(py) {
            Ok(v) => {
                if v.is_none(py) {
                    Ok(py.None())
                } else {
                    v.getattr(py, intern!(py, "first"))
                }
            }
            Err(e) => Err(e),
        }
    }

    #[getter(rest)]
    fn rest(&self, py: Python) -> PyResult<Py<PyAny>> {
        match self.seq(py) {
            Ok(v) => {
                if v.is_none(py) {
                    Ok(empty_seq(py).clone().unbind())
                } else {
                    v.getattr(py, intern!(py, "rest"))
                }
            }
            Err(e) => Err(e),
        }
    }

    #[getter(is_empty)]
    fn is_empty<'py>(&self, py: Python<'py>) -> PyResult<Borrowed<'py, 'py, PyBool>> {
        Ok(PyBool::new(py, self.seq(py)?.is_none(py)))
    }

    #[getter(is_realized)]
    fn is_realized<'py>(&self, py: Python<'py>) -> PyResult<Borrowed<'py, 'py, PyBool>> {
        let mutex = self.lock.lock();
        let state = mutex.deref().borrow();
        Ok(PyBool::new(py, matches!(*state, LazySeqState::Realized(_))))
    }

    #[pyo3(signature = (*elems))]
    fn cons<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
        elems: &Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut eiter = elems.iter();
        match eiter.nth(0) {
            Some(e) => {
                let mut l = new_py_cons(py, e, Some(slf.clone().into_bound_py_any(py)?), None)?;
                for elem in elems.iter().skip(1) {
                    l = new_py_cons(py, elem, Some(l.into_bound_py_any(py)?), None)?;
                }
                Ok(l.into_bound_py_any(py)?)
            }
            None => Ok(empty_seq(py).clone()),
        }
    }

    fn empty<'py>(&self, py: Python<'py>) -> &Bound<'py, PyAny> {
        empty_seq(py)
    }
}
