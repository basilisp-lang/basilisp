Interfaces
==========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Basilisp (like Clojure) is defined by interfaces.
All of the built-in data types are implement 0 or more of these interfaces and :lpy:ns:`basilisp.core` functions typically operate on these interfaces, rather than concrete data types (with some exceptions).

In day-to-day usage, you will not typically need to use these interfaces, but they are nevertheless helpful for understanding the abstractions Basilisp is built upon.

.. class:: basilisp.lang.interfaces.ILispObject

   Abstract base class for Lisp objects which would like to customize their ``__str__`` and Python ``__repr__`` representation.

.. lpy:currentns::  basilisp.core

.. automodule:: basilisp.lang.interfaces
   :members:
   :undoc-members:
   :exclude-members: seq_equals
   :show-inheritance:
