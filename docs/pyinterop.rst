.. _python_interop:

Python Interop
==============

Basilisp features myriad options for interfacing with host Python code.

.. _name_munging:

Name Munging
------------

Per Python's `PEP 8 naming conventions <https://www.python.org/dev/peps/pep-0008/#naming-conventions>`_, Python method and function and parameter names frequently use ``snake_case``.
Basilisp is certainly capable of reading ``snake_case`` names without any special affordance.
However, Basilisp code (like many Lisps) tends to prefer ``kebab-case`` for word separation.

Since the hyphen character used in ``kebab-case`` cannot be used in valid identifiers in Python, the Basilisp compiler automatically converts names to Python-safe identifiers before compiling.
In many of the examples below, you will see Python code referenced directly using ``kebab-case``.
When compiled, a ``kebab-case`` identifier always becomes a ``snake_case`` identifier, so calling Python code from within Basilisp blends in fairly well with standard Basilisp code.

.. note::

   The Basilisp compiler munges *all* unsafe Basilisp identifiers to safe Python identifiers, but other cases are unlikely to appear in standard Python interop usage.

.. _python_builtins:

Python Builtins
---------------

Python features a collection of `builtin <https://docs.python.org/3/library/functions.html>`_ functions which are available by default without module qualification in all Python scripts.
Python builtins are available in all Basilisp code as qualified symbols with the ``python`` namespace portion.
It is not required to import anything to enable this functionality.

::

    basilisp.user=> (python/abs -1)
    1

.. _importing_modules:

Importing Modules
-----------------

As in standard Python, it is possible to import any module importable in native Python code in Basilisp using the :lpy:fn:`import` macro, as ``(import module)``.
Submodules may be imported using the standard Python ``.`` separator: ``(import module.sub)``.

Upon import, top-level (unqualified) Python modules may be referred to using the full module name as the namespace portion of the symbol and the desired module member.
Submodules will be available under the full, dot-separated name.

To avoid name clashes from the above, you may alias imports (as in native Python code) using the same syntax as ``require``.
Both top-level modules and submodules may be aliased: ``(import [module.sub :as sm])``.

.. code-block::

    (import [os.path :as path])
    (path/exists "test.txt") ;;=> false

As with Basilisp ``refers`` (and as in Python), it is possible to refer individual module members by name into the current namespace using the ``:refer`` option.
It is also possible to refer all module members into the namespace using ``:refer :all``.

.. code-block::

   (import [math :refer [sqrt pi]])
   pi  ;; 3.141592653589793

   (import [statistics :refer :all])
   mean  ;; <function mean at 0x...>

.. warning::

   Basilisp refers names into the current module in different conceptual namespaces and resolves names against those namespaces in order of precedence, preferring Basilisp members first.
   Referred Python module members may not resolve if other names take precedence within the current namespace context.

   .. code-block::

      (import [datetime :as dt :refer :all])

      ;; This name using the module alias directly will guarantee we are referencing
      ;; the module member `datetime.time` (a class)
      dt/time  ;; <class 'datetime.time'>

      ;; ...whereas this reference prefers the `basilisp.core` function `time`
      time  ;; <function time at 0x...>

.. note::

   Users should generally prefer to use the :lpy:fn:`ns` macro for importing modules into their namespace, rather than using the :lpy:fn:`import` form directly.

   .. code-block:: clojure

      (ns myproject.ns
       (:import [os.path :as path]))

.. seealso::

   :lpy:form:`import`, :lpy:fn:`import`, :lpy:fn:`ns-imports`, :lpy:fn:`ns-import-refers`, :lpy:fn:`ns-map`

.. seealso::

   :ref:`namespaces`

.. _referencing_module_members:

Referencing Module Members
--------------------------

Once a Python module is imported into the current Namespace, it is trivial to reference module members directly.
References to Python module members appear identical to qualified Basilisp Namespace references.
Class constructors or other callables in the module can be called directly as a standard Basilisp function call.
Static members and class members can be referenced by adding the class name to the (potentially) qualified symbol namespace, separated by a single ``.``.

.. code-block:: clojure

    (import datetime)
    (datetime.datetime/now)  ;;=> #inst "2020-03-30T08:56:57.176809"

.. _accessing_object_methods_and_props:

Accessing Object Methods and Properties
---------------------------------------

Often when interfacing with native Python code, you will end up handling raw Python objects.
In such cases, you may need or want to call a method on that object or access a property.
Basilisp has specialized syntax support for calling methods on objects and accessing its properties.

To access an object's method, the :lpy:form:`.` special form can be used: ``(. object method & args)``.

.. code-block:: clojure

    (import datetime)
    (def now (datetime.datetime/now))
    (. now strftime "%Y-%m-%d")  ;;=> "2020-03-31"

As a convenience, Basilisp offers a more compact syntax for method names known at compile time: ``(.method object & args))``.

.. code-block:: clojure

    (.strftime now "%Y-%m-%d")  ;;=> "2020-03-31"

Basilisp also supports the "qualified method" syntax introduced in Clojure 1.12, albeit with fewer restrictions than the Clojure implementation.
In particular, there is no distinction between instance and static (or class) methods in syntax -- instance methods need not be prefixed with a leading ``.`` nor is it an error to prefix a static or class method with a leading ``.``.
Static and class methods typically do not take an instance of their class as the first argument, so the distinction should already be clear by usage.

.. code-block:: clojure

   ;; Python str instance method str.split()
   (python.str/split "a b c")   ;;=> #py ["a" "b" "c"]
   (python.str/.split "a b c")  ;;=> #py ["a" "b" "c"]

   ;; Python int classmethod int.from_bytes()
   (python.int/from_bytes #b"\x00\x10")   ;;=> 16
   (python.int/.from_bytes #b"\x00\x10")  ;;=> 16

In Python, objects often expose properties which can be read directly from the instance.
To read properties from the instance, you can use the ``(.- object property)`` syntax.

.. code-block:: clojure

    (.- now year)  ;;=> 2020

As with methods, Basilisp features a convenience syntax for accessing properties whose names are statically known at compile time: ``(.-property object)``.

.. code-block:: clojure

    (.-year now)  ;;=> 2020

.. note::

   Property references do not accept arguments and it is a compile-time error to pass arguments to an object property reference.

Though Basilisp generally eschews mutability, we live in a mutable world.
Many Python frameworks and libraries rely on mutable objects as part of their public API.
Methods may potentially always mutate their associated instance, but properties are often declared read-only.
For properties which are explicitly *not* read only, you can mutate their value using the ``set!`` :ref:`special form <special_forms>`.

.. code-block:: clojure

    (set! (.-property o) :new-value)  ;;=> :new-value

.. note::

   In most cases, Basilisp's method and property access features should be sufficient.
   However, in case it is not, Python's :ref:`builtins <python_builtins>` such as `getattr` and `setattr` are still available and can supplement Basilisp's interoperability features.

.. _py_interop_keyword_arguments:

Keyword Arguments
-----------------

Python functions and class constructors commonly permit callers to supply optional parameters as keyword arguments.
While Basilisp functions themselves do not *typically* expose keyword arguments, Basilisp natively supports keyword argument calls with a number of different options.
For function calls to statically known functions with a static set of keyword arguments, you can call your desired function and separate positional arguments from keyword arguments using the ``**`` special symbol.
The Basilisp compiler expects 0 or more key/value pairs (similarly to the contents of a map literal) after the ``**`` symbol in a function or method call.
It gathers all key/value pairs after that identifier, converts any keywords to valid Python identifiers (using the :ref:`name_munging` described above), and calls the Python function with those keyword arguments.

.. code-block:: clojure

    (python/open "test.txt" ** :mode "w")  ;;=> <_io.TextIOWrapper name='test.txt' mode='w' encoding='UTF-8'>

.. note::

   The symbol ``**`` does not resolve to anything in Basilisp.
   The Basilisp compiler discards it during the analysis phase of compilation.

.. note::

   It is also valid to supply keys as strings, though this is less idiomatic.
   String keys will also be munged to ensure they are valid Python identifiers.

.. _basilisp_functions_with_kwargs:

Basilisp Functions with Keyword Arguments
-----------------------------------------

In rare circumstances (such as supplying a callback function), it may be necessary for a Basilisp function to support being called with Python keyword arguments.
Basilisp can generate functions which can receive these keyword arguments and translate them into idiomatic Basilisp.
Single-arity functions and ``deftype`` methods can declare support for Python keyword arguments with the ``:kwargs`` metadata key.
Multi-arity functions and ``deftype`` methods do not support Python keyword arguments.
For functions which do support keyword arguments, two strategies are supported for generating these functions: ``:apply`` and ``:collect``.

.. note::

   Basilisp functions support a variant of :ref:`keyword_arguments` via destructuring support provided by ``fn`` and ``defn``.
   The ``:apply`` strategy relies on that style of keyword argument support to idiomatically integrate with Basilisp functions.

.. code-block:: clojure

    ^{:kwargs :apply}
    (fn [& {:as kwargs}]
      kwargs)

The ``:apply`` strategy is appropriate in situations where there are few or no positional arguments defined on your function.
With this strategy, the compiler converts the Python dict of string keys and values into a sequential stream of de-munged keyword and value pairs which are applied to the function.
As you can see in the example above, this strategy fits neatly with the existing support for :ref:`destructuring` key and value pairs from rest arguments in a function definition.

.. warning::

   With the ``:apply`` strategy, the Basilisp compiler cannot verify that the number of positional arguments matches the number defined on the receiving function, so use this strategy with caution.

.. code-block:: clojure

    ^{:kwargs :collect}
    (fn [arg1 arg2 ... {:as kwargs}]
      kwargs)

The ``:collect`` strategy is a better accompaniment to functions with positional arguments.
With this strategy, Python keyword arguments are converted into a Basilisp map with de-munged keyword arguments and passed as the final positional argument of the function.
You can use :ref:`associative_destructuring` on this final positional argument, just as you would with the map in the ``:apply`` case above.

.. _type_hinting:

Type Hinting
------------

Basilisp supports passing type hints through to the underlying generated Python using type hints by applying the ``:tag`` metadata to certain syntax elements.

In Clojure, these tags are type declarations for certain primitive types.
In Clojurescript, tags are type *hints* and they are only necessary in extremely limited circumstances to help the compiler.
In Basilisp, tags are not used by the compiler at all.
Instead, tags applied to function arguments and return values in Basilisp are applied to the underlying Python objects and are introspectable at runtime using the Python :external:py:mod:`inspect` standard library module.

Type hints may be applied to :lpy:form:`def` names, function arguments and return values, and :lpy:form:`let` local forms.

.. code-block:: clojure

   (def ^python/str s "a string")

   (defn upper
     ^python/str [^python/str s]
     (.upper s))

   (let [^python/int i 64]
     (* i 2))

.. note::

   The reader applies ``:tag`` :ref:`metadata` automatically for symbols following the ``^`` symbol, but users may manually apply ``:tag`` metadata containing any valid expression.
   Python permits any valid expression in a variable annotation, so Basilisp likewise allows any valid expression.

.. warning::

   Due to the complexity of supporting multi-arity functions in Python, only return annotations are preserved on the arity dispatch function.
   Return annotations are combined as by :external:py:obj:`typing.Union`, so ``typing.Union[str, str] == str``.
   The annotations for individual arity arguments are preserved in their compiled form, but they are challenging to access programmatically.

.. _python_slicing:

Python Slicing
--------------

Python slicing lets you extract parts of a sequence (like a list or string) using the syntax ``sequence[start:stop:step]``:

.. code-block:: python

   coll = [-3, -2, -1, 0, 1, 2, 3]
   coll[1:5:2]
   # => [-2, 0]

Basilisp provides the :lpy:fn:`basilisp.core/aslice` macro to facilitate this syntax:

.. code-block:: clojure

   (def coll #py [-3 -2 -1 0 1 2 3])
   (aslice coll 3)
   ;; => #py [-3 -2 -1]

   (aslice coll nil -3)
   ;; => #py [-3 -2 -1 0]

   (aslice coll 1 5 2)
   ;; => #py [-2 0]


This macro is just a wrapper around Python's :external:py:obj:`slice` operator combined with the :lpy:fn:`basilisp.core/aget` function:

.. code-block:: clojure

   (def coll #py [-3 -2 -1 0 1 2 3])
   (aget coll (python/slice 1 5 2))
   ;; => #py [-2 0]

.. _python_iterators:

Python Iterators
----------------

In Python, an **iterable** is an object like a list, range, or generator that can be looped over, while an **iterator** is the object that actually yields each item of the iterable one at a time using ``next()``. They are ubiquituous in Python, showing up in ``for`` loops, list comprehensions and many built in functions.

In Basilisp, iterables are treated as first-class sequences and are :lpy:fn:`basilisp.core/seq`-able, except for **single-use** iterables, which must be explicitly converted to a sequence using :lpy:fn:`basilisp.core/iterator-seq` before use.

Single-use iterables are those that return the same iterator every time one is requested.
This becomes problematic when the single-use iterable is coerced to a sequence more than once. For example:

.. code-block:: clojure

   (when (> (count iterable-coll) 0)
     (first iterable-coll))


Here, both :lpy:fn:`basilisp.core/count` and :lpy:fn:`basilisp.core/first` internally request an iterator from ``iterable-coll``.
If it is **re-iterable**, each call gets a fresh iterator beginning at the start of the collection, and the code behaves as expected.
But if it is a **single-use** iterable, like a generator, both operations share the same iterator.
As a result, ``count`` consumes all elements, and ``first`` returns ``nil``, which is wrong, since the iterator is already exhausted, leading to incorect behavior.

To prevent this subtle bug, Basilisp throws a :external:py:obj:`TypeError` when an iterator is requested from such functions.
The correct approach is to use :lpy:fn:`basilisp.core/iterator-seq` to create a sequence from it:

.. code-block:: clojure

   (let [s (iterator-seq iterable-coll)]
     (when (> (count s) 0)
       (first s)))


This ensures ``count`` and ``first`` operate on the same stable sequence rather than consuming a shared iterator.

.. _python_decorators:

Python Decorators
-----------------

.. note::

   Users wishing to apply decorators to functions are not limited to using ``:decorators`` metadata.
   This feature is provided primarily to simplify porting Python code to Basilisp.
   In Python, decorators are syntactic sugar for functions which return functions, but given the rich library of tools provided for composing functions and the ease of defining anonymous functions in Basilisp, the use of ``:decorators`` is not typically necessary in standard Basilisp code.

Python decorators are functions that modify the behavior of other functions or methods.
They are applied to a function by prefixing it with the ``@decorator_name`` syntax. A decorator takes a function as input, performs some action, and returns a new function that typically extends or alters the original function's behavior.

Basilisp offers a convenience ``:decorators`` metadata key to support Python-style decorators, which allows you to pass a vector of functions that wrap the final function emitted by the :lpy:fn:`fn` anonymous function, as well as by :lpy:fn:`defn` and its derivatives, such as :lpy:fn:`defasync`.
These decorators are applied from right to left, similar to how Python decorators work, modifying the function's behavior before it is used.

.. code-block:: clojure

    (import asyncio atexit)

    ;;; defn support
    ;;
    ;; The following will print ":goodbye!" on program exit
    (defn say-goodbye {:decorators [atexit/register]}
      []
      (println :goodbye!))

    ;;; fn support
    ;;
    ;; example decorator
    (defn add-5-decorator
      [f]
      (fn [] (+ (f) 5)))

    ;; Decorators passed to fn via form metadata
    (^{:decorators [add-5-decorator]} (fn [] 6))
    ;; => 11

    ;; Decorators passed to fn via function name metadata
    ((fn ^{:decorators [add-5-decorator]} seven [] 7))
    ;; => 12

    ;;; Decorators with arguments, and order of application (right to left)
    ;;
    ;; example decorator
    (defn mult-x-decorator
      [x]
      (fn [f]
        (fn [] (* (f) x))))

    ((fn ^{:decorators [add-5-decorator (mult-x-decorator -1)]} seven [] 7))
    ;; => -2

    ;;; defasync support
    ;;
    ;; example async decorator
    (defn add-7-async-decorator
      [f]
      ^:async (fn [] (+ (await (f)) 7)))

    (defasync ^{:decorators [add-7-async-decorator]} six
      []
      (await (asyncio/sleep 0.1))
      6)

    (asyncio/run (six))
    ;; => 13

.. _arithmetic_division:

Arithmetic Division
-------------------

.. lpy:currentns:: basilisp.core

The Python native quotient ``//`` and modulo ``%`` operators may yield different results compared to their Java counterpart's long division and modulo operators. The discrepancy arises from Python's choice of floored division (`src <http://python-history.blogspot.com/2010/08/why-pythons-integer-division-floors.html>`_, `archived <https://web.archive.org/web/20100827160949/http://python-history.blogspot.com/2010/08/why-pythons-integer-division-floors.html>`_) while Java employs truncated division for its calculations (refer to the to the `Wikipedia Modulo page <https://en.wikipedia.org/wiki/Modulo>`_ for a a comprehensive list of available division formulae).

In Clojure, the ``clojure.core/quot`` function utilizes Java's long division operator, and the ``%`` operator is employed in defining the ``clojure.core/rem`` function. The ``clojure.core/mod`` function is subsequently established through floored division based on the latter.

Basilisp has chosen to adopt the same mathematical formulae as Clojure for these three functions, rather than using the Python's built in operators under all cases. This approach offers the advantage of enhanced cross-platform compatibility without requiring modification, and ensures compatibility with examples in  `ClojureDocs <https://clojuredocs.org/>`_.

Users still have the option to use the native :external:py:func:`operator.floordiv`, i.e. Python's ``//``  operator, if they prefer so.

.. seealso::

   :lpy:fn:`quot`, :lpy:fn:`rem`, :lpy:fn:`mod`

.. _proxies:

Proxies
-------

Basilisp supports creating instances of anonymous classes deriving from one or more concrete types with the :lpy:fn:`proxy` macro.
It may be necessary to use ``proxy`` in preference to :lpy:fn:`reify` for cases when the superclass type is concrete, where ``reify`` would otherwise fail.
Proxies can also be useful in cases where it is necessary to wrap superclass methods with additional functionality or access internal state of class instances.

.. code-block::

   (def p
     (proxy [io/StringIO] []
       (write [s]
         (println "length" (count s))
         (proxy-super write s))))

   (.write p "blah")  ;; => 4
   ;; prints "length 4"
   (.getvalue p)  ;; => "blah"

.. seealso::

   :lpy:fn:`proxy`, :lpy:fn:`proxy-mappings`, :lpy:fn:`proxy-super`,
   :lpy:fn:`construct-proxy`, :lpy:fn:`init-proxy`, :lpy:fn:`update-proxy`,
   :lpy:fn:`get-proxy-class`
