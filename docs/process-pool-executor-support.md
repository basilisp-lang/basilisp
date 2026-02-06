# ProcessPoolExecutor Support in Basilisp

This document describes the implementation of `ProcessPoolExecutor` support in Basilisp, enabling true multicore parallelism for CPU-bound tasks.

## Table of Contents

1. [Background](#background)
2. [The Problem](#the-problem)
3. [The Solution](#the-solution)
4. [Implementation Details](#implementation-details)
5. [Usage](#usage)
6. [Limitations and Gotchas](#limitations-and-gotchas)
7. [Testing](#testing)

---

## Background

Basilisp provides futures for concurrent execution via the `future` macro and `future-call` function. By default, futures execute on a `ThreadPoolExecutor` bound to the dynamic var `*executor-pool*`.

However, due to Python's Global Interpreter Lock (GIL), threads cannot achieve true parallelism for CPU-bound work. The GIL ensures only one thread executes Python bytecode at a time, making `ThreadPoolExecutor` suitable only for I/O-bound tasks.

For CPU-bound parallelism, Python's `ProcessPoolExecutor` spawns separate processes, each with its own GIL, enabling true parallel execution across multiple CPU cores.

## The Problem

Before this implementation, using `ProcessPoolExecutor` with Basilisp futures failed with pickle errors:

```
AttributeError: Can't get local object 'bound_fn__STAR__.<locals>.__bound_fn__STAR____lisp_fn_2730'
```

or

```
TypeError: cannot pickle 'BasilispModule' object
```

### Root Causes

1. **`bound-fn*` creates closures**: The `future-call` function wraps submitted functions with `bound-fn*` to convey thread-local bindings. This creates an anonymous closure that standard `pickle` cannot serialize.

2. **Var objects reference Namespaces**: Thread bindings are stored as `{Var -> value}` maps. Var objects contain references to Namespace objects.

3. **Namespaces contain BasilispModule**: Each Namespace has an associated Python module (`BasilispModule`) that holds the compiled code.

4. **BasilispModule cannot be pickled**: Python's `pickle` module cannot serialize module objects, and `BasilispModule` is a subclass of `types.ModuleType`.

5. **Function globals reference modules**: Compiled Basilisp functions have `__globals__` dictionaries that reference `BasilispModule` objects (e.g., `basilisp_core`).

## The Solution

The solution involves three parts:

### 1. Use cloudpickle for Serialization

[cloudpickle](https://github.com/cloudpipe/cloudpickle) is a library that extends Python's pickle to handle closures, lambdas, and other objects that standard pickle cannot. It's used by PySpark, Dask, Ray, and other distributed computing frameworks.

cloudpickle is added as an **optional dependency** - it's only required when using `ProcessPoolExecutor`.

### 2. Make Basilisp Objects Picklable

Added `__reduce__` methods to key Basilisp types so cloudpickle can serialize them:

| Type | Serializes As | Reconstructs Via |
|------|---------------|------------------|
| `Var` | Qualified symbol (`ns/name`) | `Var.find()` or create if missing |
| `Namespace` | Name symbol | `Namespace.get_or_create()` |
| `BasilispModule` | Module name string | `importlib.import_module()` |
| `ProcessPoolExecutor` | (nothing) | Creates `ThreadPoolExecutor` |

### 3. Bootstrap Basilisp in Worker Processes

On macOS and Windows, Python's default multiprocessing start method is `spawn`, which creates fresh Python processes without any Basilisp initialization. The worker function bootstraps Basilisp before unpickling.

## Implementation Details

### Files Modified

#### `pyproject.toml`
```toml
[project.optional-dependencies]
cloudpickle = ["cloudpickle (>=2.0.0,<4.0.0)"]
```

#### `src/basilisp/lang/runtime.py`

**BasilispModule.__reduce__**:
```python
class BasilispModule(types.ModuleType):
    def __reduce__(self):
        return (_module_from_name, (self.__name__,))

def _module_from_name(name: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    return importlib.import_module(name)
```

**Var.__reduce__**:
```python
def __reduce__(self):
    qualified_sym = sym.symbol(self._name.name, ns=self._ns.name)
    return (_var_from_symbol, (qualified_sym,))

def _var_from_symbol(qualified_sym: sym.Symbol) -> Var:
    v = Var.find(qualified_sym)
    if v is not None:
        return v
    # Create the Var if it doesn't exist (for user-defined vars in subprocesses)
    ns = Namespace.get_or_create(sym.symbol(qualified_sym.ns))
    return Var.intern(ns, sym.symbol(qualified_sym.name),
                      Var._Var__UNBOUND_SENTINEL, dynamic=True)
```

**Namespace.__reduce__**:
```python
def __reduce__(self):
    return (_namespace_from_symbol, (self._name,))

def _namespace_from_symbol(name: sym.Symbol) -> Namespace:
    return Namespace.get_or_create(name)
```

#### `src/basilisp/lang/futures.py`

**ProcessPoolExecutor.submit** - uses cloudpickle:
```python
def submit(self, fn, *args, **kwargs):
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
```

**ProcessPoolExecutor.__reduce__** - unpickles as ThreadPoolExecutor:
```python
def __reduce__(self):
    # In subprocess, nested futures should use threads, not more processes
    return (_create_thread_pool_executor, ())
```

**Worker function** - bootstraps Basilisp:
```python
def _execute_cloudpickled_fn(pickled_fn: bytes, pickled_args: bytes):
    from basilisp.main import init
    init()  # Bootstrap Basilisp in subprocess

    fn = cloudpickle.loads(pickled_fn)
    args, kwargs = cloudpickle.loads(pickled_args)
    return fn(*args, **kwargs)
```

## Usage

### Installation

```bash
pip install basilisp[cloudpickle]
```

### Basic Example

```clojure
(defn cpu-work [n]
  (python/sum (python/range n)))

(binding [*executor-pool* (basilisp.lang.futures/ProcessPoolExecutor)]
  (let [futures [(future (cpu-work 10000000))
                 (future (cpu-work 10000000))
                 (future (cpu-work 10000000))
                 (future (cpu-work 10000000))]
        results (mapv deref futures)]
    (println "Results:" results)))
```

### Using pmap

```clojure
(binding [*executor-pool* (basilisp.lang.futures/ProcessPoolExecutor)]
  (doall (pmap cpu-work [10000000 10000000 10000000 10000000])))
```

## Limitations and Gotchas

### 1. Only Explicitly Bound Vars Are Conveyed

**Gotcha**: Only vars that are explicitly rebound in a `binding` form are conveyed to subprocesses. Root values of dynamic vars are NOT conveyed.

```clojure
(def ^:dynamic *my-var* 10)

;; WRONG - *my-var* is not bound, only has root value
(binding [*executor-pool* (ProcessPoolExecutor)]
  (future *my-var*))  ;; ERROR: Unable to resolve symbol

;; CORRECT - explicitly bind *my-var*
(binding [*executor-pool* (ProcessPoolExecutor)
          *my-var* 10]
  (future *my-var*))  ;; Works!
```

**Why**: `bound-fn*` only captures thread-local bindings (set via `binding`), not root values. In the subprocess, the var doesn't exist unless it was in the bindings map.

### 2. User-Defined Functions Must Be Serializable

**Gotcha**: Functions that close over non-serializable objects will fail.

```clojure
;; This works - closes over a simple value
(let [x 5]
  (binding [*executor-pool* (ProcessPoolExecutor)]
    @(future (* x 2))))  ;; => 10

;; This might fail - if `some-obj` isn't picklable
(let [some-obj (create-unpicklable-thing)]
  (binding [*executor-pool* (ProcessPoolExecutor)]
    @(future (.method some-obj))))
```

### 3. Subprocess Startup Overhead

**Gotcha**: Each subprocess must bootstrap Basilisp, adding ~1-2 seconds overhead on first task.

```clojure
;; First batch of tasks will be slower due to process startup + bootstrap
(binding [*executor-pool* (ProcessPoolExecutor)]
  (time (doall (pmap cpu-work [1000000 1000000]))))
;; "Elapsed time: 2500 msecs" (includes startup)

;; Subsequent batches reuse warm processes
(binding [*executor-pool* (ProcessPoolExecutor)]
  (time (doall (pmap cpu-work [1000000 1000000]))))
;; "Elapsed time: 200 msecs" (processes already warm)
```

**Mitigation**: Reuse the same `ProcessPoolExecutor` instance for multiple batches of work.

### 4. Nested Futures Use ThreadPoolExecutor

**Gotcha**: If your task creates more futures inside the subprocess, they'll use `ThreadPoolExecutor`, not `ProcessPoolExecutor`.

```clojure
(defn task-with-nested-futures []
  ;; This nested future runs on ThreadPoolExecutor in the subprocess
  @(future (+ 1 2)))

(binding [*executor-pool* (ProcessPoolExecutor)]
  @(future (task-with-nested-futures)))
```

**Why**: `ProcessPoolExecutor` unpickles as `ThreadPoolExecutor` to avoid spawning processes from within processes, which can cause issues.

### 5. No Shared State Between Processes

**Gotcha**: Each process has its own memory space. Atoms, refs, and other mutable state are NOT shared.

```clojure
(def counter (atom 0))

(binding [*executor-pool* (ProcessPoolExecutor)]
  (doall (pmap (fn [_] (swap! counter inc)) (range 4))))

@counter  ;; Still 0! Each process had its own copy of the atom
```

**Solution**: Use return values to communicate results, or use proper IPC mechanisms if you need shared state.

### 6. Print Output May Be Interleaved

**Gotcha**: Output from multiple processes may be interleaved unpredictably.

```clojure
(binding [*executor-pool* (ProcessPoolExecutor)]
  (doall (pmap #(println "Processing" %) (range 4))))
;; Output order is not guaranteed
```

### 7. Exceptions in Subprocesses

**Gotcha**: Exceptions in subprocess tasks are re-raised when you `deref` the future, but the stack trace points to the subprocess.

```clojure
(binding [*executor-pool* (ProcessPoolExecutor)]
  @(future (throw (ex-info "Oops" {}))))
;; Raises the exception with subprocess stack trace
```

### 8. Large Data Transfer Overhead

**Gotcha**: Arguments and return values are serialized/deserialized. Large data structures add overhead.

```clojure
;; Slow - serializes large-vector 4 times
(binding [*executor-pool* (ProcessPoolExecutor)]
  (doall (pmap process-fn [large-vector large-vector large-vector large-vector])))

;; Better - each task generates its own data
(binding [*executor-pool* (ProcessPoolExecutor)]
  (doall (pmap (fn [seed] (process-fn (generate-data seed))) [1 2 3 4])))
```

## Testing

Run the ProcessPoolExecutor tests:

```bash
pytest tests/basilisp/core/test_futures.lpy -v -k "process_pool"
```

Or run manually:

```clojure
(import time os)

(defn cpu-work [n]
  (println (str "PID " (os/getpid) " processing"))
  (python/sum (python/range n)))

(binding [*executor-pool* (basilisp.lang.futures/ProcessPoolExecutor)]
  (let [start (time/perf-counter)
        futures [(future (cpu-work 5000000))
                 (future (cpu-work 5000000))
                 (future (cpu-work 5000000))
                 (future (cpu-work 5000000))]
        results (mapv deref futures)
        elapsed (- (time/perf-counter) start)]
    (println "Results:" results)
    (println (str "Total time: " (format "%.2f" elapsed) "s"))))
```

Expected output shows 4 different PIDs and parallel execution time.
