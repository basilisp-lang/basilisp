.. _differences_from_clojure:

Differences from Clojure
========================

Basilisp strives to be roughly compatible with Clojure, but just as ClojureScript `diverges from Clojure <https://clojurescript.org/about/differences>`_ at points, so too does Basilisp.
Being a hosted language like Clojure (which celebrates its host, rather than hiding it) means that certain host-specific constructs cannot be replicated on every platform.
We have tried to replicate the behavior of Clojure as closely as we can while still staying true to Python.

This document outlines the major differences between the two implementations so users of both can understand where Basilisp differs and adjust their code accordingly.
If a feature differs between the two implementations and it is not stated here, please first check if there is an open `issue on GitHub <https://github.com/basilisp-lang/basilisp/issues>`_ to implement or align the feature with Clojure or to clarify if it should be omitted.

.. _hosted_on_python:

Hosted on Python
----------------

Unlike Clojure, Basilisp is hosted on the Python VM.
Basilisp supports versions of Python 3.8+.
Basilisp projects and libraries may both import Python code and be imported by Python code (once the Basilisp runtime has been :ref:`initialized <bootstrapping>` and the import hooks have been installed).

.. _type_differences:

Type Differences
----------------

* ``nil`` corresponds to Python's ``None``\.
* Python does not offer different integer sizes, so ``short``, ``int``, and ``long`` are identical.
* Python does not offer different precision floating point numbers, so ``double`` and ``float`` are identical.
* Type coercions generally delegate to the relevant Python constructor, which handles such things natively.
* Collections

  * Sorted sets, sorted maps, and array maps are not implemented (support is tracked in `#416 <https://github.com/basilisp-lang/basilisp/issues/416>`_).

.. _arithmetic_comparison:

Arithmetic Comparison
---------------------

Basilisp, in contrast to Clojure, does not distinguish between integer (``int``) and floating point (``float``) as `separate categories for equality comparison purposes <https://clojure.org/guides/equality>`_ where the ``=`` comparison between any ``int`` and ``float`` returns ``false``. Instead, it adopts Python's ``=`` comparison operator semantics, where the ``int`` is optimistically converted to a ``float`` before the comparison. However, beware that this conversion can lead to `certain caveats in comparison <https://stackoverflow.com/a/30100743>`_ where in rare cases seemingly exact ``int`` and ``float`` numbers may still compare to ``false`` due to limitations in floating point number representation.

In Clojure, this optimistic equality comparison is performed by the ``==`` function. In Basilisp, ``==`` is aliased to behave the same as ``=``.

.. note::

   Basilisp's ``=`` will perform as expected when using Python `Decimal <https://docs.python.org/3/library/decimal.html>`__ typed :ref:`floating_point_numbers`.

.. seealso::

   Python's `floating point arithmetic <https://docs.python.org/3/tutorial/floatingpoint.html>`_ documentation

.. _concurrent_programming:

Concurrent Programming
----------------------

Python is famous for it's `Global Interpreter Lock <https://docs.python.org/3/glossary.html#term-global-interpreter-lock>`_ limiting performance in the multi-core case.
As such, users may call into question the value of Clojure's concurrency-focused primitives in a single-threaded context.
However, ClojureScript's own `"Differences from Clojure" <https://clojurescript.org/about/differences>`_ document puts its best:

   Clojure’s model of values, state, identity, and time is valuable even in single-threaded environments.

That said, there are some fundamental differences and omissions in Basilisp that make it differ from Clojure.

* Atoms work just as in Clojure.
* Basilisp does not include Ref types or software transactional memory (STM) support.
* Basilisp does not include Agent support (support is tracked in `#413 <https://github.com/basilisp-lang/basilisp/issues/413>`_).
* All Vars are reified at runtime and users may use the :lpy:fn:`binding` macro as in Clojure.

  * Non-dynamic Vars are compiled into Python variables and references to those Vars are made using Python variables using :ref:`direct_linking`.
  * Vars are created in all cases, but only used in certain cases.

.. _reader_differences:

Reader
------

* :ref:`Numbers <numeric_literals>`

  * Python integers natively support unlimited precision, so there is no difference between regular integers and those suffixed with ``N`` (which are read as ``BigInt``\s in Clojure).
  * Floating point numbers are read as Python ``float``\s by default and subject to the limitations of that type on the current Python VM.
    Floating point numbers suffixed with ``M`` are read as Python `Decimal <https://docs.python.org/3/library/decimal.html#decimal.Decimal>`_ types and support user-defined precision.
  * Ratios are supported and are read in as Python `Fraction <https://docs.python.org/3/library/fractions.html#fractions.Fraction>`_ types.
  * Python natively supports Complex numbers.
    The reader will return a complex number for any integer or floating point literal suffixed with ``J``.

* :ref:`Characters <character_literals>`

  * Python does not support character types, so characters are returned as single-character strings.

* :ref:`Python data types <data_readers>`

  * The reader will return the native Python data type corresponding to the Clojure type in functionality if the value is prefixed with ``#py``.

.. _regular_expressions:

Regular Expressions
-------------------

Basilisp regular expressions use Python's `regular expression <https://docs.python.org/3/library/re.html>`_ syntax and engine.

.. _repl_differences:

REPL
----

Basilisp's REPL experience closely matches that of Clojure's.

.. _evaluation_differences:

Evaluation
----------

Basilisp code has the same evaluation semantics as Clojure.
The :lpy:fn:`load` and :lpy:fn:`load-file` functions are supported though their usage is generally discouraged.
Basilisp does not perform any locals clearing.

.. _special_form_differences:

Special Forms
-------------

Basilisp special forms should be identical to their Clojure counterparts unless otherwise noted below.

* :lpy:form:`def` does not support the ``^:const`` metadata key.
* :lpy:form:`if` does not use any boxing behavior as that is not relevant for Python.
* The JVM specific ``locking``\, ``monitor-enter``\, and ``monitor-exit`` special forms are not implemented.
* The Python VM specific :lpy:form:`await` and :lpy:form:`yield` forms are included to support Python interoperability.

.. _namespace_differences:

Namespaces
----------

Basilisp namespaces are reified at runtime and support the full set of ``clojure.core`` namespace APIs.
Namespaces correspond to a single Python `module <https://docs.python.org/3/library/sys.html#sys.modules>`_ which is where the compiled code (essentially anything that has been :lpy:form:`def`\-ed) lives.
Users should rarely need to be concerned with this implementation detail.

As in Clojure, namespaces are bootstrapped using the :lpy:fn:`ns` header macro at the top of a code file.
There are some differences between ``ns`` in Clojure and ``ns`` in Basilisp:

* Users may use ``:refer-basilisp`` and ``:refer-clojure`` interchangeably to control which of the :lpy:ns:`basilisp.core` functions are referred into the new namespace.
* Prefix lists are not supported for any of the import or require selectors.
* Automatic namespace aliasing: if a namespaces starting with ``clojure.`` is required and does not exist, but a corresponding namespace starting with ``basilisp.`` does exist, Basilisp will import the latter automatically with the former as an alias.

.. _lib_differences:

Libs
----

Support for Clojure libs is `planned <https://github.com/basilisp-lang/basilisp/issues/668>`_\.

.. _core_lib_differences:

basilisp.core
-------------

- :lpy:fn:`basilisp.core/int` coerces its argument to an integer. When given a string input, Basilisp will try to interpret it as a base 10 number, whereas in Clojure, it will return its ASCII/Unicode index if it is a character (or fail if it is a string). Use `lpy:fn:`ord` instead to return the character index if required.

- :lpy:fn:`basilisp.core/float` coerces its argument to a floating-point number. When given a string input, Basilisp will try to parse it as a floating-point number, whereas Clojure will raise an error if the input is a character or a string.

.. _refs_and_transactions_differences:

Refs and Transactions
---------------------

Neither refs nor transactions are supported.

.. _agents_differences:

Agents
------

Agents are not currently supported. Support is tracked in `#413 <https://github.com/basilisp-lang/basilisp/issues/413>`_.

.. _host_interop_differences:

Host Interop
------------

Host interoperability features generally match those of Clojure.

* :lpy:fn:`new` is a macro for Clojure compatibility, as the ``new`` keyword is not required for constructing new objects in Python.
* `Python builtins <https://docs.python.org/3/library/functions.html>`_ are available under the special namespace ``python`` (as ``python/abs``, for instance) without requiring an import.

.. seealso::

   :ref:`python_interop`

.. _type_hinting_differences:

Type Hinting
^^^^^^^^^^^^

Type hints may be applied anywhere they are supported in Clojure (as the ``:tag`` metadata key), but the compiler does not currently use them for any purpose.
Tags provided for ``def`` names, function arguments and return values, and :lpy:form:`let` locals will be applied to the resulting Python AST by the compiler wherever possible.
Particularly in the case of function arguments and return values, these tags maybe introspected from the Python `inspect <https://docs.python.org/3/library/inspect.html>`_ module.
There is no need for type hints anywhere in Basilisp right now, however.

.. _compilation_differences:

Compilation
-----------

Basilisp's compilation is intended to work more like Clojure's than ClojureScript's, in the sense that code is meant to be JIT compiled from Lisp code into Python code at runtime.
Basilisp compiles namespaces into modules one form at a time, which brings along all of the attendant benefits (macros can be defined and immediately used) and drawbacks (being unable to optimize code across the entire namespace).
``gen-class`` is not required or implemented in Basilisp, but :lpy:fn:`gen-interface` is.
Users may still create dynamic classes using Python's ``type`` builtin, just as they could do in Python code.

.. seealso::

   :ref:`compiler`

.. _core_libraries_differences:

Core Libraries
--------------

Basilisp includes ports of some of the standard libraries from Clojure which should generally match the source in functionality.

* :lpy:ns:`basilisp.data` is a port of ``clojure.data``
* :lpy:ns:`basilisp.edn` is a port of ``clojure.edn``
* :lpy:ns:`basilisp.io` is a port of ``clojure.java.io``
* :lpy:ns:`basilisp.set` is a port of ``clojure.set``
* :lpy:ns:`basilisp.shell` is a port of ``clojure.java.shell``
* :lpy:ns:`basilisp.string` is a port of ``clojure.string``
* :lpy:ns:`basilisp.test` is a port of ``clojure.test``
* :lpy:ns:`basilisp.walk` is a port of ``clojure.walk``
