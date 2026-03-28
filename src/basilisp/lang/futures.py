import atexit
import sys
import weakref
from collections.abc import Callable
from concurrent.futures import Future as _Future  # noqa # pylint: disable=unused-import
from concurrent.futures import process as _cf_process
from concurrent.futures import ProcessPoolExecutor as _ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
from concurrent.futures import TimeoutError as _TimeoutError
from typing import Any, TypeVar

import attr
from typing_extensions import ParamSpec

from basilisp.lang.interfaces import IBlockingDeref, IPending

try:
    import cloudpickle

    _CLOUDPICKLE_AVAILABLE = True
except ImportError:
    _CLOUDPICKLE_AVAILABLE = False

T = TypeVar("T")
P = ParamSpec("P")


@attr.frozen(eq=True, repr=False)
class Future(IBlockingDeref[T], IPending):
    _future: "_Future[T]"

    def __repr__(self):  # pragma: no cover
        return self._future.__repr__()

    def cancel(self) -> bool:
        return self._future.cancel()

    def cancelled(self) -> bool:
        return self._future.cancelled()

    def deref(
        self, timeout: float | None = None, timeout_val: T | None = None
    ) -> T | None:
        try:
            return self._future.result(timeout=timeout)
        except _TimeoutError:
            return timeout_val

    def done(self) -> bool:
        return self._future.done()

    @property
    def is_realized(self) -> bool:
        return self.done()

    # Pass `Future.result(timeout=...)` through so `Executor.map(...)` can
    # still work with this Future wrapper.
    def result(self, timeout: float | None = None) -> T:
        return self._future.result(timeout=timeout)


# Basilisp's standard Future executor is the `ThreadPoolExecutor`, but since
# it is set via a dynamic variable, it can be rebound using the binding macro.
# Callers may wish to use a process pool if they have CPU bound work.


def _execute_cloudpickled_fn(pickled_fn: bytes, pickled_args: bytes) -> Any:
    """Execute a cloudpickle-serialized function in a worker process.

    This function is called in the child process and must be a top-level
    function (not a closure) so it can be pickled by the standard pickle module.

    The function bootstraps Basilisp in the subprocess before unpickling since
    the subprocess may not have Basilisp initialized (especially on macOS/Windows
    where 'spawn' is the default multiprocessing start method).
    """
    # Import here to avoid circular imports at module load time
    from basilisp.main import init
    from basilisp import importer

    # Bootstrap Basilisp in the subprocess - this is idempotent
    init()

    # Ensure the Basilisp import hook is active in the child process.
    if not any(isinstance(o, importer.BasilispImporter) for o in sys.meta_path):
        importer.hook_imports()

    fn = cloudpickle.loads(pickled_fn)
    args, kwargs = cloudpickle.loads(pickled_args)
    return fn(*args, **kwargs)


def _create_thread_pool_executor() -> "ThreadPoolExecutor":
    """Create a ThreadPoolExecutor for use in worker processes.

    This is used when unpickling a ProcessPoolExecutor - in a subprocess,
    nested futures should use threads rather than spawning more processes.
    """
    return ThreadPoolExecutor()


def _shutdown_executor(executor_ref: "weakref.ReferenceType[ProcessPoolExecutor]") -> None:
    """Best-effort shutdown helper for a ProcessPoolExecutor weakref."""
    executor = executor_ref()
    if executor is None:
        return
    try:
        executor.shutdown()
    except Exception:
        pass


class ProcessPoolExecutor(_ProcessPoolExecutor):
    """A ProcessPoolExecutor that uses cloudpickle to serialize closures.

    This executor wraps Python's ProcessPoolExecutor and uses cloudpickle
    (when available) to serialize functions, enabling the use of closures
    and locally-defined functions that standard pickle cannot handle.

    If cloudpickle is not installed, a RuntimeError will be raised when
    attempting to submit work. Install it with: pip install basilisp[cloudpickle]

    The executor supports the context manager protocol:
        with ProcessPoolExecutor() as executor:
            future = executor.submit(fn, *args)

    When used as a context manager, cleanup happens automatically. Otherwise,
    shutdown() will be called at program exit via atexit. If auto_shutdown is
    True (default), the executor is also shut down when unbound from
    *executor-pool*.
    """

    def __init__(self, max_workers: int | None = None, auto_shutdown: bool = True):
        # Python's ProcessPoolExecutor performs a system limits check that can
        # raise PermissionError in constrained environments (e.g. sandboxed
        # macOS runners). Temporarily patch the check to allow execution.
        orig_check = getattr(_cf_process, "_check_system_limits", None)
        if orig_check is None:
            super().__init__(max_workers=max_workers)
        else:
            def _safe_check_system_limits():
                try:
                    orig_check()
                except PermissionError:
                    return None

            _cf_process._check_system_limits = _safe_check_system_limits
            try:
                super().__init__(max_workers=max_workers)
            finally:
                _cf_process._check_system_limits = orig_check
        self._auto_shutdown = auto_shutdown
        # Register shutdown to be called on program exit to prevent resource leaks
        # This will be unregistered if the executor is used as a context manager
        self._atexit_registered = True
        executor_ref = weakref.ref(self)
        self._atexit_ref = executor_ref

        def _atexit_shutdown() -> None:
            _shutdown_executor(executor_ref)

        self._atexit_shutdown = _atexit_shutdown
        atexit.register(_atexit_shutdown)

    def __enter__(self):
        # Delegate to parent's __enter__
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Unregister atexit handler since we're cleaning up via context manager
        if self._atexit_registered:
            atexit.unregister(self._atexit_shutdown)
            self._atexit_registered = False
        # Delegate to parent's __exit__ which calls shutdown()
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __del__(self):  # pragma: no cover - best-effort cleanup
        try:
            if getattr(self, "_atexit_registered", False):
                try:
                    atexit.unregister(self._atexit_shutdown)
                except Exception:
                    pass
                self._atexit_registered = False
            self.shutdown()
        except Exception:
            pass

    def __reduce__(self):
        """Enable pickling of ProcessPoolExecutor.

        When bound-fn* captures *executor-pool* and it's a ProcessPoolExecutor,
        the executor needs to be picklable. In the subprocess, we substitute
        a ThreadPoolExecutor since nested futures should use threads rather
        than spawning additional processes.
        """
        return (_create_thread_pool_executor, ())

    # pylint: disable=arguments-differ
    def submit(  # type: ignore[override]
        self, fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> "Future[T]":
        if not _CLOUDPICKLE_AVAILABLE:
            raise RuntimeError(
                "cloudpickle is required for ProcessPoolExecutor. "
                "Install it with: pip install basilisp[cloudpickle]"
            )
        pickled_fn = cloudpickle.dumps(fn)
        pickled_args = cloudpickle.dumps((args, kwargs))
        return Future(
            super().submit(_execute_cloudpickled_fn, pickled_fn, pickled_args)
        )


class ThreadPoolExecutor(_ThreadPoolExecutor):
    def __init__(
        self,
        max_workers: int | None = None,
        thread_name_prefix: str = "basilisp-futures",
    ):
        super().__init__(max_workers=max_workers, thread_name_prefix=thread_name_prefix)
        # Register shutdown to be called on program exit to prevent resource leaks
        atexit.register(self.shutdown)

    # pylint: disable=arguments-differ
    def submit(  # type: ignore[override]
        self, fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> "Future[T]":
        return Future(super().submit(fn, *args, **kwargs))
