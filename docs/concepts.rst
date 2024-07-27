.. _concepts:

Concepts
========

.. lpy:currentns:: basilisp.core

.. _data_structures:

Data Structures
---------------

TBD

.. _seqs:

Seqs
----

TBD

.. seealso::

   :lpy:fn:`lazy-seq`, :lpy:fn:`seq`, :lpy:fn:`first`, :lpy:fn:`rest`, :lpy:fn:`next`, :lpy:fn:`second`, :lpy:fn:`seq?`, :lpy:fn:`nfirst`, :lpy:fn:`fnext`, :lpy:fn:`nnext`, :lpy:fn:`empty?`, :lpy:fn:`seq?`, :py:class:`basilisp.lang.interfaces.ISeq`

.. _destructuring:

Destructuring
-------------

The most common type of name binding encountered in Basilisp code is that of a single symbol to a value.
For example, below the name ``a`` is bound to the result of the expression ``(+ 1 2)``::

   (let [a (+ 1 2)]
     a)

In many cases this form of name binding is sufficient.
However, when dealing with data nested in vectors or maps of known shapes, it would be much more convenient to bind those values directly without needing to write collection accessor functions by hand.
Basilisp supports a form of name binding known as destructuring, which allows convenient name binding of values from within sequential and associative data structures.
Destructuring is supported everywhere names are bound: :lpy:form:`fn` argument vectors, :lpy:form:`let` bindings, and :lpy:form:`loop` bindings.

.. note::

   Names without a corresponding element in the data structure (typically due to absence) will bind to ``nil``.

.. seealso::

   :lpy:fn:`destructure`

.. _sequential_destructuring:

Sequential Destructuring
^^^^^^^^^^^^^^^^^^^^^^^^

Sequential destructuring is used to bind values from sequential types.
The binding form for sequential destructuring is a vector.
Names in the vector will be bound to their corresponding indexed element in the sequential expression value, fetched from that type as by :lpy:fn:`nth`.
As a result, any data type supported by ``nth`` natively supports sequential destructuring, including vectors, lists, strings, Python lists, and Python tuples.
It is possible to collect the remaining unbound elements as a ``seq`` by providing a trailing name separated from the individual bindings by an ``&``.
The rest element will be bound as by :lpy:fn:`nthnext`.
It is also possible to bind the full collection to a name by adding a trailing ``:as`` name after all binding forms and optional rest binding.

.. code-block::

   (let [[a b c & others :as coll] [:a :b :c :d :e :f]]
     [a b c others coll])
   ;;=> [:a :b :c (:d :e :f) [:a :b :c :d :e :f]]

Sequential destructuring may also be nested:

.. code-block::

   (let [[[a b c] & others :as coll] [[:a :b :c] :d :e :f]]
     [a b c others coll])
   ;;=> [:a :b :c (:d :e :f) [[:a :b :c] :d :e :f]]

.. _associative_destructuring:

Associative Destructuring
^^^^^^^^^^^^^^^^^^^^^^^^^

Associative destructuring is used to bind values from associative types.
The binding form for associative destructuring is a map.
Names in the map will be bound to their corresponding key in the associative expression value, fetched from that type as by :lpy:fn:`get`.
Asd a result, any associative types supported by ``get`` natively supports sequential destructuring, including maps, vectors, strings, sets, and Python dicts.
It is possible to bind the full collection to a name by adding an ``:as`` key.
Default values can be provided for keys by providing a map of binding names to default values using the ``:or`` key.

.. code-block::

   (defn f [{x :a y :b :as m :or {y 18}}]
     [x y m])

   (f {:a 1 :b 2})  ;;=> [1 2 {:a 1 :b 2}]
   (f {:a 1})       ;;=> [1 18 {:a 1}]
   (f {})           ;;=> [nil 18 {}]

For the common case where the names you intend to bind directly match the corresponding keyword name, you can use the ``:keys`` notation.

.. code-block::

   (defn f [{:keys [a b] :as m}]
     [a b m])

   (f {:a 1 :b 2})  ;;=> [1 2 {:a 1 :b 2}]
   (f {:a 1})       ;;=> [1 nil {:a 1}]
   (f {})           ;;=> [nil nil {}]

There exists a corresponding construct for the symbol and string key cases as well: ``:syms`` and ``:strs``, respectively.

.. code-block::

   (defn f [{:strs [a] :syms [b] :as m}]
     [a b m])

   (f {"a" 1 'b 2})  ;;=> [1 2 {"a" 1 'b 2}]

.. note::

   The keys for the ``:strs`` construct must be convertible to valid Basilisp symbols.

It is possible to bind namespaced keys directly using either namespaced individual keys or a namespaced version of ``:keys`` as ``:ns/keys``.
Values will be bound to the symbol by their *name* only (as by :lpy:fn:`name`) -- the namespace is only used for lookup in the associative data structure.

.. code-block::

   (let [{a :a b :a/b :c/keys [c d]} {:a   "a"
                                      :b   "b"
                                      :a/a "aa"
                                      :a/b "bb"
                                      :c/c "cc"
                                      :c/d "dd"}]
     [a b c d])
   ;;=> ["a" "bb" "cc" "dd"]

.. _keyword_arguments:

Keyword Arguments
^^^^^^^^^^^^^^^^^

Basilisp functions can be defined with support for keyword arguments by defining the "rest" argument in an :lpy:fn:`defn` or :lpy:fn:`fn` form with associative destructuring.
Callers can pass interleaved key/value pairs as positional arguments to the function and they will be collected into a single map argument which can be destructured.
If a single trailing map argument is passed by callers (instead of or in addition to other key/value pairs), that value will be joined into the final map.

.. code-block::

   (defn f [& {:keys [a b] :as kwargs}]
     [a b kwargs])

   (f :a 1 :b 2)    ;;=> [1 2 {:a 1 :b 2}]
   (f :a 1 {:b 2})  ;;=> [1 2 {:a 1 :b 2}]
   (f {:a 1 :b 2})  ;;=> [1 2 {:a 1 :b 2}]

.. note::

   Basilisp keyword arguments are distinct from Python keyword arguments.
   Basilisp functions can be :ref:`defined with Python compatible keyword arguments <basilisp_functions_with_kwargs>` but the style described here is intended primarily for Basilisp functions called only by other Basilisp functions.

.. warning::

   The trailing map passed to functions accepting keyword arguments will silently overwrite values passed positionally.
   Callers should take care when using the trailing map calling convention.

   .. code-block::

      (defn f [& {:keys [a b] :as kwargs}]
        [a b kwargs])

      (f :a 1 {:b 2 :a 3})
      ;;=> [3 2 {:a 3 :b 2}]

.. _nested_destructuring:

Nested Destructuring
^^^^^^^^^^^^^^^^^^^^

Both associative and sequential destructuring binding forms may be nested within one another.

.. code-block::

   (let [[{:keys [a] [e f] :d} [b c]] [{:a 1 :d [4 5]} [:b :c]]]
     [a b c e f])
   ;;=> [1 :b :c 4 5]

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

.. _metadata:

Metadata
--------

TBD

.. seealso::

   :ref:`Reading metadata on literals <reader_metadata>`, :lpy:fn:`meta`, :lpy:fn:`with-meta`, :lpy:fn:`vary-meta`, :lpy:fn:`alter-meta!`, :lpy:fn:`reset-meta!`

.. _delays:

Delays
------

Delays are containers for deferring expensive computations until such time as the result is needed.
Create a new delay with the :lpy:fn:`delay` macro.
Results will not be computed until you attempt to :lpy:fn:`deref` or :lpy:fn:`force` evaluation.
Once a delay has been evaluated, it caches its results and returns the cached results on subsequent accesses.

.. code-block::

   basilisp.user=> (def d (delay (println "evaluating") (+ 1 2 3)))
   #'basilisp.user/d

   basilisp.user=> d
   <basilisp.lang.delay.Delay object at 0x1077803a0>

   basilisp.user=> (force d)
   evaluating
   6

   basilisp.user=> (force d)
   6

.. seealso::

   :lpy:fn:`delay`, :lpy:fn:`delay?`, :lpy:fn:`force`, :lpy:fn:`realized?`, :lpy:fn:`deref`

.. _promises:

Promises
--------

Promises are containers for receiving a deferred result, typically from another thread.
The value of a promise can be written exactly once using :lpy:fn:`deliver`.
Threads may await the results of the promise using a blocking :lpy:fn:`deref` call.

.. seealso::

   :lpy:fn:`promise`, :lpy:fn:`deliver`, :lpy:fn:`realized?`, :lpy:fn:`deref`

.. _atoms:

Atoms
-----

TBD

.. seealso::

   :lpy:fn:`atom`, :lpy:fn:`compare-and-set!`, :lpy:fn:`reset!`, :lpy:fn:`reset-vals!`, :lpy:fn:`swap!`, :lpy:fn:`swap-vals!`, :lpy:fn:`deref`, :ref:`references_and_refs`

.. _references_and_refs:

References and Refs
-------------------

TBD

.. seealso::

   :lpy:fn:`alter-meta!`, :lpy:fn:`reset-meta!`, :lpy:fn:`add-watch`, :lpy:fn:`remove-watch`, :lpy:fn:`get-validator`, :lpy:fn:`set-validator!`

.. _transients:

Transients
----------

TBD

.. seealso::

   :lpy:fn:`transient`, :lpy:fn:`persistent!`, :lpy:fn:`assoc!`, :lpy:fn:`conj!`, :lpy:fn:`disj!`, :lpy:fn:`dissoc!`, :lpy:fn:`pop!`

.. _volatiles:

Volatiles
---------

TBD

.. seealso::

   :lpy:fn:`volatile!`, :lpy:fn:`volatile?`, :lpy:fn:`vreset!`, :lpy:fn:`vswap!`

.. _transducers:

Transducers
-----------

TBD

.. seealso::

   :lpy:fn:`eduction`, :lpy:fn:`completing`, :lpy:fn:`halt-when`, :lpy:fn:`sequence`, :lpy:fn:`transduce`, :lpy:fn:`into`, :lpy:fn:`cat`

.. _hierarchies:

Hierarchies
-----------

TBD

.. seealso::

   :lpy:fn:`make-hierarchy`, :lpy:fn:`ancestors`, :lpy:fn:`descendents`, :lpy:fn:`parents`, :lpy:fn:`isa?`, :lpy:fn:`derive`, :lpy:fn:`underive`

.. _multimethods:

Multimethods
------------

TBD

.. seealso::

   :lpy:fn:`defmulti`, :lpy:fn:`defmethod`, :lpy:fn:`methods`, :lpy:fn:`get-method`, :lpy:fn:`prefer-method`, :lpy:fn:`prefers`, :lpy:fn:`remove-method`, :lpy:fn:`remove-all-methods`

.. _protocols:

Protocols
---------

TBD

.. seealso::

   :lpy:fn:`defprotocol`, :lpy:fn:`protocol?`, :lpy:fn:`extend`, :lpy:fn:`extend-protocol`, :lpy:fn:`extend-type`, :lpy:fn:`extenders`, :lpy:fn:`extends?`, :lpy:fn:`satisfies?`

.. _data_types:

Data Types
----------

TBD

.. seealso::

   :lpy:fn:`deftype`

.. _records:

Records
-------

TBD

.. seealso::

   :ref:`records` , :lpy:fn:`defrecord` , :lpy:fn:`record?`