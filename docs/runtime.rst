.. _runtime:

Runtime
=======

.. lpy:currentns:: basilisp.core

The Basilisp runtime is vast and comprised of many different moving parts.
The primary components that users will interface with, however, are Namespaces and Vars.

.. _namespaces:

Namespaces
----------

Namespaces are the primary unit of code organization provided by Basilisp.
To the Basilisp runtime, a Namespace is simply a mapping of names to objects (objects, strings, etc.); this mapping is known as a :ref:`Var <vars>`.
In practice, Namespaces are typically written and organized as individual files in the filesystem.
Namespaces may be nested to produce a deeper source tree where that makes sense to do.

Users create Namespaces using the :lpy:fn:`basilisp.core/ns` macro at the top of every code file.

.. code-block:: clojure

   (ns myproject.ns
     "This namespace does x, y, and z."
     (:require
      [basilisp.string :as str])
     (:import datetime))

Under the hood, this magic macro will do a bunch of convenient setup that users would otherwise have to do themselves.
First, it will refer the public contents of :lpy:ns:`basilisp.core`, allowing unqualified references to all functions in that namespace anywhere within the new namespace.
Secondly, it will require the namespace :lpy:ns:`basilisp.string` which makes the functions from that namespace available in the current namespace using the shortened prefix ``str``.
That means within the newly created namespace you will be able to refer to :lpy:fn:`basilisp.string/alpha?` as simply ``str/alpha?``.
Afterwards, the ``ns`` macro will import the Python module `datetime <https://docs.python.org/3/library/datetime.html>`_, which we can use in a similar way, referring to objects as ``datetime/date`` for instance.
Finally, ``ns`` will set the dynamic Var :lpy:var:`basilisp.core/*ns*` to ``myproject.ns``, which informs the compiler that any new Vars or functions defined right now should be associated with ``myproject.ns``, not any other namespace.

.. note::

   This documentation frequently conflates namespaces and physical files.
   It is not necessary for a namespace to be defined in only one file, nor is it necessary that one file contain only one namespace.
   However, while these practices are technically permitted, they are generally discouraged since it makes it harder to figure out where code is defined and organized.

.. seealso::

   :lpy:fn:`all-ns`, :lpy:fn:`create-ns`, :lpy:fn:`find-ns`, :lpy:fn:`intern`, :lpy:fn:`ns`, :lpy:fn:`ns-name`, :lpy:fn:`ns-resolve`, :lpy:fn:`remove-ns`, :lpy:fn:`resolve`, :lpy:fn:`the-ns`

.. _namespace_requires:

Requires
^^^^^^^^

.. note::

   :lpy:fn:`require` and the ``(:require ...)`` form of :lpy:fn:`ns` are the preferred methods for requiring namespaces and referring Vars.
   :lpy:fn:`refer` and :lpy:fn:`use` are both older, more limited functions which ``refer`` has subsumed and they are only included for Clojure compatibility.

.. seealso::

   :lpy:fn:`ns-aliases`, :lpy:fn:`ns-interns`, :lpy:fn:`ns-map`, :lpy:fn:`ns-publics`, :lpy:fn:`ns-refers`, :lpy:fn:`ns-unalias`, :lpy:fn:`ns-unmap`, :lpy:fn:`refer`, :lpy:fn:`require`, :lpy:fn:`use`

.. _namespace_imports:

Imports
^^^^^^^

.. seealso::

   :lpy:form:`import`, :lpy:fn:`import`, :lpy:fn:`ns-imports`, :lpy:fn:`ns-map`

.. _vars:

Vars
----

Vars are mutable :ref:`reference types <references_and_refs>` which hold a reference to something.
Users typically interact with Vars with the :lpy:form:`def` form and the :lpy:fn:`basilisp.core/defn` macro which create Vars to hold he result of the expression or function.
All values created with these forms are stored in Vars and interned in a Namespace so they can be looked up later.
The Basilisp compiler uses Vars interned in Namespaces during name resolution to determine if a name is referring to a local name (perhaps in a :lpy:form:`let` binding or as a function argument) or if it refers to a Var.

Vars may have metadata, which generally originates on the ``name`` symbol given during a :lpy:form:`def`.
Specific metadata keys given during the creation of a Var can enable specific features that may be useful for some Vars.

.. seealso::

   :lpy:fn:`alter-var-root`, :lpy:fn:`find-var`, :lpy:fn:`thread-bound?`, :lpy:fn:`var-get`, :lpy:fn:`var-set`

.. _dynamic_vars:

Dynamic Vars
^^^^^^^^^^^^

Vars created with the ``^:dynamic`` metadata key are known as "dynamic" Vars.
Dynamic Vars include a thread-local stack of value bindings that can be overridden using the :lpy:fn:`basilisp.core/binding` macro.
This may be a suitable alternative to requiring users to pass in an infrequently changing value as an argument to your function.
Basilisp uses this in :lpy:ns:`basilisp.core` with things such as :lpy:var:`*in*`

.. note::

   Dynamic Vars are typically named with so-called "earmuffs" (leading and trailing ``*`` characters) to indicate their dynamic nature.
   For instance, if you were going to call the Var ``dynamic-var``, you'd actually name it ``*dynamic-var*``.

.. note::

   Dynamic Vars are never :ref:`direct linked <direct_linking>`, so they are always subject to Var indirection.
   Users should be aware of this limitation when using dynamic Vars in hot paths.

.. _private_vars:

Private Vars
^^^^^^^^^^^^

Vars created with the ``^:private`` metadata key are considered "private" within a namespace and access to those Vars from other namespaces is limited.
Private Vars are not included by any :ref:`require or refer <namespace_requires>` operations and may not be referenced by using the fully-qualified symbol name of the Var either.

This is typically useful for cases where you might want to define an implementation function which you do not want to expose or export as a public API.

.. warning::

   Since private Vars are not accessible outside of the namespace they are defined in, callers should take care not to use them in macro definitions since they will result in compile-time errors for users of the macro.

.. _unbound_vars:

Unbound Vars
^^^^^^^^^^^^

Vars defined without a value (as by ``(def some-var)``) are considered "unbound".
Such Vars a root value defined which is different from ``nil`` and which only compares equal to itself and other unbound values referencing the same Var.