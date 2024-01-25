.. _python_interop:

Python Interop
==============

Basilisp features myriad options for interfacing with host Python code.

.. _name_munging:

Name Munging
------------

Per Python's `PEP 8 naming conventions <https://www.python.org/dev/peps/pep-0008/#naming-conventions>`_, Python method and function names frequently use ``snake_case``.
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
Note that none of the other convenience features or flags from :lpy:fn:`require` are available, so you will not be able to, say, refer unqualified module members into the current Namespace.

.. code-block::

    (import [os.path :as path])
    (path/exists "test.txt") ;;=> false

.. note::

   Users should generally prefer to use the :lpy:fn:`ns` macro for importing modules into their namespace, rather than using the :lpy:fn:`import` form directly.

   .. code-block:: clojure

      (ns myproject.ns
       (:import [os.path :as path]))

.. warning::

   Unlike in Python, imported module names and aliases cannot be referred to directly in Basilisp code.
   Module and Namespace names are resolved separately from local names and will not resolve as unqualified names.

.. seealso::

   :lpy:form:`import`, :lpy:fn:`import`, :lpy:fn:`ns-imports`, :lpy:fn:`ns-map`

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
As you can see in the example above, this strategy fits neatly with the existing support for destructuring key and value pairs from rest arguments in a function definition.

.. warning::

   With the ``:apply`` strategy, the Basilisp compiler cannot verify that the number of positional arguments matches the number defined on the receiving function, so use this strategy with caution.

.. code-block:: clojure

    ^{:kwargs :collect}
    (fn [arg1 arg2 ... {:as kwargs}]
      kwargs)

The ``:collect`` strategy is a better accompaniment to functions with positional arguments.
With this strategy, Python keyword arguments are converted into a Basilisp map with de-munged keyword arguments and passed as the final positional argument of the function.
You can use map destructuring on this final positional argument, just as you would with the map in the ``:apply`` case above.

Type Hinting
------------

Basilisp supports passing type hints through to the underlying generated Python using type hints by applying the ``:tag`` metadata to certain syntax elements.

In Clojure, these tags are type declarations for certain primitive types.
In Clojurescript, tags are type *hints* and they are only necessary in extremely limited circumstances to help the compiler.
In Basilisp, tags are not used by the compiler at all.
Instead, tags applied to function arguments and return values in Basilisp are applied to the underlying Python objects and are introspectable at runtime using the Python `inspect <https://docs.python.org/3/library/inspect.html>`_ standard library module.

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
   Return annotations are combined as by ``typing.Union``, so ``typing.Union[str, str] == str``.
   The annotations for individual arity arguments are preserved in their compiled form, but they are challenging to access programmatically.