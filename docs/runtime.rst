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
Namespace names should match the document's place in the source tree, with forward slash (``/``) characters being replaced by periods (``.``) and underscores (``_``) replaces with hyphens (``-``).

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
Afterwards, the ``ns`` macro will import the Python module :external:py:mod:`datetime`, which we can use in a similar way, referring to objects as ``datetime/date`` for instance.
Finally, ``ns`` will set the dynamic Var :lpy:var:`basilisp.core/*ns*` to ``myproject.ns``, which informs the compiler that any new Vars or functions defined right now should be associated with ``myproject.ns``, not any other namespace.

.. note::

   This documentation frequently conflates namespaces and physical files for simplicity's sake.
   It is not necessary for a namespace to be defined in only one file, nor is it necessary that one file contain only one namespace.
   However, while these practices are technically permitted, they are generally discouraged since it makes it harder to figure out where code is defined and organized.

.. seealso::

   :lpy:fn:`all-ns`, :lpy:fn:`create-ns`, :lpy:fn:`find-ns`, :lpy:fn:`intern`, :lpy:fn:`ns`, :lpy:fn:`ns-name`, :lpy:fn:`ns-resolve`, :lpy:fn:`remove-ns`, :lpy:fn:`resolve`, :lpy:fn:`the-ns`

.. seealso::

   :ref:`importing_modules`

.. _namespace_requires:

Requires
^^^^^^^^

Requires are the primary way users establish linkages between different namespaces, similarly to how Python's ``import`` statement connects different Python modules and packages.
In typical usage, namespaces are required in the :lpy:fn:`ns` preamble at the top of every code file.
In a REPL context, it may make sense to use the :lpy:fn:`require` function to require a namespace in an ad-hoc fashion during development.
Both tools generally provide the same set of capabilities, as the ``ns`` form ultimately compiles down to calls to the function ``require``.

Required namespaces may be required using their full name, aliased to a shorter name (such as ``str`` for ``basilisp.string``), or even have individual Vars from the namespace referred in and reference as if they were defined in the same namespace.
Users may even combine the options together.
When using the ``:refer`` feature of ``require``, users may also choose to instead refer all Vars from the target namespace by using the ``:all`` keyword in place of the vector of Var names to require.

.. code-block:: clojure

   (ns myproject.ns
     (:require
      [basilisp.string :as str]
      [basilisp.io]
      [basilisp.edn :as edn :refer [read-string]]
      [basilisp.set :refer :all]
      [basilisp.walk :refer [walk]]))

.. warning::

   Referring ``:all`` Vars from another namespace is generally discouraged, since it can clog up the namespace with potentially unused names and can make it challenging for readers to figure out where a Var came from.

As noted above in :ref:`namespaces`, the :lpy:fn:`ns` macro performs an implicit ``[basilisp.core :refer :all]`` by default, allowing users to refer to all core functions without qualification.
In general this is desirable, since you will be interacting with :lpy:ns:`basilisp.core` a lot.
However, in some cases, you may wish to suppress certain Vars from being referred, particularly if you are defining Vars with clashing names.
In such cases, you can instruct the ``ns`` macro to exclude specific Vars from ``basilisp.core``:

.. code-block::

   (ns myproject.ns
     (:refer-basilisp :exclude [get]))

There are other filtering and selection criteria which can be included on both ``:refer-basilisp`` and ``:require`` sections of the ``ns`` macro.
See the documentation for :lpy:fn:`require` for more details.

.. note::

   :lpy:fn:`require` and the ``(:require ...)`` form of :lpy:fn:`ns` are the preferred methods for requiring namespaces and referring Vars.

   :lpy:fn:`refer` and :lpy:fn:`use` are both older, more limited functions which ``refer`` has subsumed and they are only included for Clojure compatibility.

.. seealso::

   :lpy:fn:`ns-aliases`, :lpy:fn:`ns-interns`, :lpy:fn:`ns-map`, :lpy:fn:`ns-publics`, :lpy:fn:`ns-refers`, :lpy:fn:`ns-unalias`, :lpy:fn:`ns-unmap`, :lpy:fn:`refer`, :lpy:fn:`require`, :lpy:fn:`use`

.. _vars:

Vars
----

Vars are mutable :ref:`reference types <reference_types>` which hold a reference to something.
Users typically interact with Vars with the :lpy:form:`def` form and the :lpy:fn:`basilisp.core/defn` macro which create Vars to hold he result of the expression or function.
All values created with these forms are stored in Vars and interned in a Namespace so they can be looked up later.
The Basilisp compiler uses Vars interned in Namespaces during name resolution to determine if a name is referring to a local name (perhaps in a :lpy:form:`let` binding or as a function argument) or if it refers to a Var.

Vars may have metadata, which generally originates on the ``name`` symbol given during a :lpy:form:`def`.
Specific metadata keys given during the creation of a Var can enable specific features that may be useful for some Vars.

.. seealso::

   :lpy:fn:`alter-var-root`, :lpy:fn:`find-var`, :lpy:fn:`thread-bound?`, :lpy:fn:`var-get`, :lpy:fn:`var-set`, :lpy:fn:`with-redefs`, :lpy:fn:`with-redefs-fn`

.. _var_metadata:

Metadata
^^^^^^^^

Whenever a Var is defined as by :lpy:form:`def`, the compiler typically adds some metadata about where the Var was defined.
The following is a non-exhaustive list of potential metadata keys that may be set by the compiler.

All Vars might get the following keys:

- ``:ns`` the :ref:`namespace <namespaces>` the Var is interned in
- ``:name`` the name of the Var as a symbol
- ``:file`` the name of the source file where the Var was defined, or if it was defined in a REPL or via a string (such as by :lpy:fn:`eval`) then a descriptive string surrounded by ``<...>``
- ``:line``, ``:col``, ``:end-line``, ``:end-col`` location metadata about where in ``:file`` the Var was defined
- ``:doc`` the docstring provided at the time the Var was interned, if one
- ``:tag`` typically a return value for functions or type hint for values

Users may provide the following metadata which the compiler will pass through:

- ``:redef`` (see :ref:`compiler` for more details)
- ``:private`` (see :ref:`private_vars` below)
- ``:dynamic`` (see :ref:`dynamic_vars` below)

Vars containing functions (typically defined via some variant of :lpy:fn:`defn` or :lpy:fn:`defmacro`) might get the following keys:

- ``:macro`` if the function is a macro and eligible to be called during macroexpansion
- ``:arglists`` a sequence of the argument vectors for each defined arity of the function

.. seealso::

   :ref:`reference_types`

.. _dynamic_vars:

Dynamic Vars
^^^^^^^^^^^^

Vars created with the ``^:dynamic`` metadata key are known as "dynamic" Vars.
Dynamic Vars include a thread-local stack of value bindings that can be overridden using the :lpy:fn:`basilisp.core/binding` macro.
This may be a suitable alternative to requiring users to pass in an infrequently changing value as an argument to your function.
Basilisp uses this in :lpy:ns:`basilisp.core` with things such as :lpy:var:`*in*`, :lpy:var:`*out*`, and :lpy:var:`*data-readers*`.

For example, if you wanted to fetch all of the data being printed to ``*out*`` as a string, you could trivially do so with this construct:

.. code-block:: clojure

   (import io)
   (let [s (io/StringIO)]
     (binding [*out* s]
       ...)
     (.getvalue s))

Note that this functionality already exists as :lpy:fn:`with-out-str`, but it serves as a good example of how to use :lpy:fn:`binding` with a dynamic Var.

.. note::

   Dynamic Vars are typically named with so-called "earmuffs" (leading and trailing ``*`` characters) to indicate their dynamic nature.
   For instance, if you were going to call the Var ``dynamic-var``, you'd actually name it ``*dynamic-var*``.

.. note::

   Dynamic Vars are never :ref:`direct linked <direct_linking>`, so they are always subject to Var indirection.
   Users should be aware of this limitation when using dynamic Vars in hot paths.

.. seealso::

   :lpy:fn:`binding`


.. _binding_conveyance:

Binding Conveyance
##################

Basilisp supports the concept of "binding conveyance" which allows copying the active set of dynamic Var bindings in the current thread when submitting work to another thread.
Both :lpy:fn:`future` and :lpy:fn:`pmap` support this feature natively.

.. seealso::

    :lpy:fn:`bound-fn`, :lpy:fn:`bound-fn*`, :lpy:fn:`get-thread-bindings`, :lpy:fn:`pop-thread-bindings`, :lpy:fn:`push-thread-bindings`, :lpy:fn:`with-bindings`, :lpy:fn:`with-bindings*`

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