.. _concepts:

Concepts
========

.. lpy:currentns:: basilisp.core

.. _seqs:

Seqs
----

.. _macros:

Macros
------

Like many Lisps, Basilisp supports extending its syntax using macros.
Macros are created using the :lpy:fn:`defmacro` macro in :lpy:ns:`basilisp.core`.
Syntax for the macro usage generally matches that of the sibling :lpy:fn:`defn` macro, should be a relatively easy transition.

Once a macro is defined, it is immediately available to the compiler.
You may define a macro and then use it in the next form!

The primary difference between a macro and a standard function is that macros are evaluated *at compile* time and they receive unevaluated expressions, whereas functions are evaluated *at runtime* and arguments will be fully evaluated before being passed to the function.
Macros should return the unevaluated replacement code that should be compiled.
Code returned by macros *must be legal code* -- symbols must be resolvable, functions must have the correct number of arguments, maps must have keys and corresponding values, etc.

Macros created with ``defmacro`` automatically have access to two additional parameters (which *should not* be listed in the macro argument list): ``&env`` and ``&form``.
``&form`` contains the original unevaluated form (including the invocation of the macro itself).
``&env`` contains a mapping of all symbols available to the compiler at the time of macro invocation -- the values are maps representing the binding AST node.

.. note::

   Being able to extend the syntax of your language using macros is a powerful feature.
   However, with great power comes great responsibility.
   Introducing new and unusual syntax to a language can make it harder to onboard new developers and can make code harder to reason about.
   Before reaching for macros, ask yourself if the problem can be solved using standard functions first.

.. warning::

   Macro writers should take care not to emit any references to :ref:`private_vars` in their macros, as these will not resolve for users outside of the namespace they are defined in, causing compile-time errors.

.. seealso::

   :ref:`syntax_quoting`, :lpy:form:`quote`, :lpy:fn:`gensym`, :lpy:fn:`macroexpand`, :lpy:fn:`macroexpand-1`, :lpy:fn:`unquote`, :lpy:fn:`unquote-splicing`

.. _binding_conveyance:

Binding Conveyance
------------------

TBD

.. _destructuring:

Destructuring
-------------

TBD

.. _references_and_refs:

References and Refs
-------------------

TBD

.. _transducers:

Transducers
-----------

TBD

.. _hierarchies:

Hierarchies
-----------

TBD

.. _multimethods:

Multimethods
------------

TBD

.. _protocols:

Protocols
---------

TBD

.. _data_types:

Data Types
----------

TBD

.. _records:

Records
-------

TBD