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
Basilisp supports versions of Python 3.6+.
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

TBD

.. _evaluation_differences:

Evaluation
----------

TBD

.. _special_form_differences:

Special Forms
-------------

TBD

.. _namespace_differences:

Namespaces
----------

TBD


.. _var_differences:

Vars
----

TBD

.. _library_differences:

Libraries
---------

TBD

.. _host_interop_differences:

Host Interop
------------

TBD

.. _type_hinting_differences:

Type Hinting
^^^^^^^^^^^^

Type hints are supported, but unused by the compiler in any capacity.

.. _compilation_differences:

Compilation
-----------

TBD

.. _core_libraries_differences:

Core Libraries
--------------

Basilisp includes ports of some of the standard libraries from Clojure which should generally match the source in functionality.

* :lpy:ns:`basilisp.data` is a port of ``clojure.data``
* :lpy:ns:`basilisp.set` is a port of ``clojure.set``
* :lpy:ns:`basilisp.string` is a port of ``clojure.string``
* :lpy:ns:`basilisp.test` is a port of ``clojure.test``
* :lpy:ns:`basilisp.walk` is a port of ``clojure.walk``