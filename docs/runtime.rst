.. _runtime:

Runtime
=======

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
First, it will require the namespace :lpy:ns:`basilisp.string` which makes the functions from that namespace available in the current namespace using the shortened prefix ``str``.
That means within the newly created namespace you will be able to refer to :lpy:fn:`basilisp.string/alpha?` as simply ``str/alpha?``.
Secondly, the ``ns`` macro will import the Python module `datetime <https://docs.python.org/3/library/datetime.html>`_, which we can use in a similar way, referring to objects as ``datetime/date`` for instance.
Finally, ``ns`` will set the dynamic Var :lpy:var:`basilisp.core/*current-ns*` to ``myproject.ns``, which informs the compiler that any new Vars or functions defined right now should be associated with ``myproject.ns``, not any other namespace.

.. note::

   This documentation frequently conflates namespaces and physical files.
   It is not necessary for a namespace to be defined in only one file, nor is it necessary that one file contain only one namespace.
   However, while these practices are technically permitted, they are generally discouraged since it makes it harder to figure out where code is defined and organized.

.. _vars:

Vars
----

Vars are mutable boxes which hold a reference to something.
Users typically interact with Vars with the :lpy:form:`def` form and the :lpy:fn:`basilisp.core/defn` macro which create Vars to hold he result of the expression or function.
All values created with these forms are stored in Vars and interned in a Namespace so they can be looked up later.
The Basilisp compiler uses Vars interned in Namespaces during name resolution to determine if a name is referring to a local name (perhaps in a :lpy:form:`let` or function argument) or if it refers to a Var.

Vars may have metadata, which generally originates on the ``name`` symbol given during a :lpy:form:`def`.
Specific metadata keys given during the creation of a Var can enable specific features that may be useful for some Vars.

.. _dynamic_vars:

Dynamic Vars
^^^^^^^^^^^^

Vars created with the ``^:dynamic`` metadata key are known as "dynamic" Vars.
Dynamic Vars are typically named with so-called "earmuffs" (leading and trailing ``*`` characters) to indicate their dynamic nature.
For instance, if you were going to call the Var ``dynamic-var``, you'd actually name it ``*dynamic-var*``.

Dynamic Vars include a thread-local stack of value bindings that can be overridden using the :lpy:fn:`basilisp.core/binding` macro.

TBD

.. _private_vars:

Private Vars
^^^^^^^^^^^^

TBD

.. _unbound_vars:

Unbound Vars
^^^^^^^^^^^^

TBD