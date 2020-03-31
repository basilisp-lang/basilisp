.. _python_interop:

Python Interop
==============

Basilisp features myriad options for interfacing with host Python code.

.. contents:: Python Interop
   :depth: 2

.. _name_munging:

Name Munging
------------

Per Python's `PEP 8 naming conventions <https://www.python.org/dev/peps/pep-0008/#naming-conventions>`, Python method and function names frequently use ``snake_case``.
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

Python features a collection of `builtin <https://docs.python.org/3.8/library/functions.html>` functions which are available by default without module qualification in all Python scripts.
Python builtins are available in all Basilisp code as qualified symbols with the ``python`` namespace portion.
It is not required to import anything to enable this functionality.

::

    basilisp.user=> (python/abs -1)
    1

.. _importing_modules:

Importing Modules
-----------------

As in standard Python, it is possible to import any module importable in native Python code in Basilisp using the ``(import module)`` macro.
Submodules may be imported using the standard Python ``.`` separator: ``(import module.sub)``.

Upon import, top-level (unqualified) Python modules may be referred to using the full module name as the namespace portion of the symbol and the desired module member.
**Unlike in Python,** Submodules will be available under the name of *first* dot-separated name segment (which is usually the top-most module in the hierarchy).

To avoid name clashes from the above, you may alias imports (as in native Python code) using the same syntax as ``require``.
Both top-level modules and submodules may be aliased: ``(import [module.sub :as sm])``.
Note that none of the other convenience features or flags from ``require`` are available, so you will not be able to, say, refer unqualified module members into the current Namespace.

.. warning::

   Unlike in Python, imported module names and aliases cannot be referred to directly in Basilisp code.
   Module and Namespace names are resolved separately from local names and will not resolve as unqualified names.

.. code-block::

    (import [os.path :as path])
    (path/exists "test.txt") ;;=> false

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

To access an object's method, the ``.`` special form can be used: ``(. object method & args)``.

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
For properties which are explicitly *not* read only, you can mutate their value using the ``set!`` :ref:`special_forms`.

.. code-block:: clojure

    (set! (.-property o) :new-value)  ;;=> :new-value

.. _keyword_arguments:

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
Functions can declare support for Python keyword arguments with the ``:kwargs`` metadata key.
Two strategies are supported for generating these functions: ``:apply`` and ``:collect``.

.. note::

   Basilisp functions support a variant of keyword arguments via destructuring support provided by ``fn`` and ``defn``.
   The ``:apply`` strategy relies on that style of keyword argument support to idiomatically integrate with Basilisp functions.

.. code-block:: clojure

    ^{:kwargs :apply}
    (fn [& {:as kwargs}]
      kwargs)

The ``:apply`` strategy is appropriate in situations where there are few or no positional arguments defined on your function.
With this strategy, the compiler converts the Python dict of string keys and values into a sequentual stream of de-munged keyword and value pairs which are applied to the function.
As you can see in the example above, this strategy fits neatly with the existing support for destructuring key and value pairs from rest arguments in a function definition.

.. warning::

   With the ``:apply`` strategy, the Basilisp compiler cannot verify that the number of positional arguments matches the number defined on the receiving function.

.. code-block:: clojure

    ^{:kwargs :collect}
    (fn [arg1 arg2 ... {:as kwargs}]
      kwargs)

The ``:collect`` strategy is a better accompaniment to functions with positional arguments.
With this strategy, Python keyword arguments are converted into a Basilisp map with de-munged keyword arguments and passed as the final positional argument of the function.
You can use map destructuring on this final positional argument, just as you would with the map in the ``:apply`` case above.
