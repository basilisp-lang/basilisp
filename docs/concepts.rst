.. _concepts:

Concepts
========

.. lpy:currentns:: basilisp.core

This document outlines some of the key high-level concepts of Basilisp, most of which behave identically to their Clojure counterparts.
The sections below tend to focus on how each of these concepts can be applied while using Basilisp.
For those looking for more of a philosophical discussion of each of these concepts, you may find the corresponding `Clojure documentation <https://clojure.org/reference>`_ enlightening as it frequently emphasizes the *motivations* for the various concepts more heavily than this documentation.

.. _data_structures:

Data Structures
---------------

Basilisp provides a comprehensive set of immutable data structures and a set of scalar types inherited from the host Python environment.

.. _nil:

nil
^^^

The value ``nil`` corresponds to Python's ``None``.
Any type is potentially ``nil``, though many Basilisp functions are intended to handle ``nil`` values gracefully.
Only the value ``nil`` and the :ref:`boolean <boolean_values>` value ``false`` are considered logical false in conditions.

.. seealso::

   :lpy:fn:`nil?`

.. _boolean_values:

Boolean Values
^^^^^^^^^^^^^^

The values ``true`` and ``false`` correspond to Python's ``True`` and ``False``, respectively.
They are singleton instances of :external:py:class:`bool`
Only the values ``nil`` and ``false`` are considered logical false in conditions.

.. seealso::

   :lpy:fn:`false?`, :lpy:fn:`true?`

.. _numbers:

Numbers
^^^^^^^

Basilisp exposes all of the built-in numeric types from Python and operations with those values match Python unless otherwise noted.

Integral values correspond to Python's arbitrary precision :external:py:class:`int` type.

Floating point values are represented by :external:py:class:`float`.
For fixed-point arithmetic with user specified precision (corresponding with Clojure's ``BigDecimal`` type float suffixed with an ``M``), Basilisp uses Python's :external:py:class:`decimal.Decimal` class.

Complex numbers are backed by Python's :external:py:class:`complex`.

Ratios are represented by Python's :external:py:class:`fractions.Fraction` type.

.. seealso::

   :ref:`arithmetic_division`

.. _strings_and_byte_strings:

Strings and Byte Strings
^^^^^^^^^^^^^^^^^^^^^^^^

Basilisp exposes both Python's base :external:py:class:`str` and :external:py:class:`bytes` types.

.. note::

   Basilisp does not have a first class character type since there is no equivalent in Python.
   :ref:`reader_character_literals` can be read from source code, but will be converted into single-character strings.

.. seealso::

   :lpy:ns:`basilisp.string` for an idiomatic string manipulation library

.. _keywords:

Keywords
^^^^^^^^

Keywords are symbolic identifiers which always evaluate to themselves.
Keywords consist of a name and an optional namespace, both of which are strings.
The textual representation of a keyword includes a single leading ``:``, which is not part of the name or namespace.

Keywords are also functions of one or 2 arguments, roughly equivalent to calling :lpy:fn:`get` on a map or set with an optional default value argument.
If the first argument is a :ref:`map <maps>`, then looks up the value associated with the keyword in the map.
If the first argument is a :ref:`set <sets>`, then looks up if the keyword is a member of the set and returns itself if so.
Returns the default value or ``nil`` (if no default value is specified) if either check fails.

.. code-block::

   (def m {:kw 1 :other 2})
   (:kw m)            ;; => 1
   (get m :kw)        ;; => 1
   (:some-kw m)       ;; => nil
   (:some-kw m 3)     ;; => 3
   (get m :some-kw 3) ;; => 3

.. note::

   Keyword values are interned and keywords are compared by identity, not by value.

.. warning::

   Keywords can be created programmatically via :lpy:fn:`keyword` which may not be able to be read back by the :ref:`reader`, so use caution when creating keywords programmatically.

.. seealso::

   :lpy:fn:`keyword`, :lpy:fn:`name`, :lpy:fn:`namespace`, :lpy:fn:`keyword?`

.. _symbols:

Symbols
^^^^^^^

Symbols are symbolic identifiers which are typically used to refer to something else.
Symbols consist of a name and an optional namespace, both strings.

Symbols, like :ref:`keywords`, can also be called like a function similar to :lpy:fn:`get` on a map or set with an optional default value argument.
If the first argument is a :ref:`map <maps>`, then looks up the value associated with the symbol in the map.
If the first argument is a :ref:`set <sets>`, then looks up if the symbol is a member of the set and returns itself if so.
Returns the default value or ``nil`` (if no default value is specified) if either check fails.

.. code-block::

   (def m {'sym 1 'other 2})
   ('sym m)            ;; => 1
   (get m 'sym)        ;; => 1
   ('some-sym m)       ;; => nil
   ('some-sym m 3)     ;; => 3
   (get m 'sym 3)      ;; => 3

.. note::

   Basilisp will always try to resolve unquoted symbols, so be sure to wrap symbols in as ``(quote sym)`` or ``'sym`` if you just want a symbol.

.. warning::

   Symbols can be created programmatically via :lpy:fn:`symbol` which may not be able to be read back by the :ref:`reader`, so use caution when creating symbols programmatically.

.. seealso::

   :lpy:fn:`symbol`, :lpy:fn:`name`, :lpy:fn:`namespace`, :lpy:fn:`gensym`, :lpy:form:`quote`

.. _collection_types:

Collection Types
^^^^^^^^^^^^^^^^

Basilisp includes the following data structures, all of which are both immutable and persistent.
APIs which "modify" collections in fact produce new collections which may or may not share some structure with the original collection.
As a result of their immutability, all of these collections are thread-safe.

Many of Basilisp's built-in collection types support creating :ref:`transient <transients>` versions of themselves for more efficient modification in a tight loop.

.. seealso::

   :lpy:fn:`count`, :lpy:fn:`conj`, :lpy:fn:`seq`

.. _lists:

Lists
#####

Lists are singly-linked lists.
Unlike most other Basilisp collections, Lists directly implement :py:class:`basilisp.lang.interfaces.ISeq` (see :ref:`seqs`).
You can get the count of a list in ``O(n)`` time via :lpy:fn:`count`.
Items added via :lpy:fn:`conj` are added to the front of the list.

.. seealso::

   :lpy:fn:`list`, :lpy:fn:`peek`, :lpy:fn:`pop`, :lpy:fn:`list?`

.. _queues:

Queues
######

Queues are doubly-linked lists.
You get the count of a queue in ``O(1)`` time via :lpy:fn:`count`.
Items added via :lpy:fn:`conj` are added to the end of the queue.

.. seealso::

   :lpy:fn:`queue`, :lpy:fn:`peek`, :lpy:fn:`pop`, :lpy:fn:`queue?`

.. _vectors:

Vectors
#######

Vectors are sequential collections much more similar to Python lists or arrays in other languages.
Vectors return their count in ``O(1)`` time via :lpy:fn:`count`.
:lpy:fn:`conj` adds items to the end of a vector.
Random access to vector elements by index (via :lpy:fn:`get` or :lpy:fn:`nth`) is ``O(log32(n))``.
You can reverse a vector in constant time using :lpy:fn:`rseq`.

Vectors be called like a function similar to :lpy:fn:`nth` with an index and an optional default value, returning the value at the specified index if found.
Returns the default value or ``nil`` (if no default value is specified) otherwise.

.. code-block::

   (def v [:a :b :c])
   (v 0)                ;; => :a
   (v 5)                ;; => nil
   (v 5 :g)             ;; => :g

.. seealso::

   :lpy:fn:`vector`, :lpy:fn:`vec`, :lpy:fn:`get`, :lpy:fn:`nth`, :lpy:fn:`peek`, :lpy:fn:`pop`, :lpy:fn:`rseq`, :lpy:fn:`vector?`

.. _maps:

Maps
####

Maps are unordered, associative collections which map arbitrary keys to values.
Keys must be hashable.
Maps return their count in ``O(1)`` time via :lpy:fn:`count`.
Random access to map values is ``O(log(n))``.

:lpy:fn:`conj` accepts any of the following types, adding new keys or replacing keys as appropriate:

- Another map; values will be merged in from left to right with keys from the rightmost map taking precedence in the instance of a conflict
- A map entry
- 2 element vector; the first element will be treated as the key and the second the value

Calling :lpy:fn:`seq` on a map yields successive map entries, which are roughly equivalent to 2 element vectors.

Maps be called like a function similar to :lpy:fn:`get` with a key and an optional default value, returning the value at the specified key if found.
Returns the default value or ``nil`` (if no default value is specified) otherwise.

.. code-block::

   (def m {:a 0 :b 1})
   (m :a)               ;; => 0
   (m :g)               ;; => nil
   (m :g 5)             ;; => 5

.. seealso::

   :lpy:fn:`hash-map`, :lpy:fn:`assoc`, :lpy:fn:`assoc-in`, :lpy:fn:`get`, :lpy:fn:`get-in`, :lpy:fn:`find`, :lpy:fn:`update`, :lpy:fn:`update-in`, :lpy:fn:`dissoc`, :lpy:fn:`merge`, :lpy:fn:`merge-with`, :lpy:fn:`map-entry`, :lpy:fn:`key`, :lpy:fn:`val`, :lpy:fn:`keys`, :lpy:fn:`vals`, :lpy:fn:`select-keys`, :lpy:fn:`update-keys`, :lpy:fn:`update-vals`, :lpy:fn:`map?`

.. _sets:

Sets
####

Sets are unordered groups of unique values.
Values must be hashable.
Sets return their count in ``O(1)`` time via :lpy:fn:`count.`

Sets be called like a function similar to :lpy:fn:`get` with a key and an optional default value, returning the value if it exists in the set.
Returns the default value or ``nil`` (if no default value is specified) otherwise.

.. code-block::

   (def s #{:a :b :c})
   (s :a)                ;; => :a
   (s :g)                ;; => nil
   (s :g :g)             ;; => :g

.. seealso::

   :lpy:fn:`hash-set`, :lpy:fn:`set`, :lpy:fn:`disj`, :lpy:fn:`contains?`, :lpy:fn:`set?`

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

Basilisp symbols and collection types support optional metadata.
As the name implies, metadata describes the data contained in a collection or the symbol.
Users will most frequently encounter metadata used either as a hint for the compiler or as an artifact added to a symbol after compilation.
However, metadata is reified at runtime and available for use for purposes other than compiler hints.

.. note::

   Metadata is not considered when comparing two objects for equality or when generating their hash codes.

.. note::

   Despite the fact that metadata is not considered for object equality, object metadata is nevertheless immutably linked to the object.
   Changing the metadata of an object as by :lpy:fn:`with-meta` or :lpy:fn:`vary-meta` will result in a different object.

.. code-block::

   (def m ^:kw ^python/str ^{:map :yes} {:data []})

   ;; will emit compiler metadata since we're inspecting the metadata of the Var
   (meta #'m)                                         ;; => {:end-col 48 :ns basilisp.user :end-line 1 :col 0 :file "<REPL Input>" :line 1 :name m}

   ;; will emit the metadata we created when we def'ed m
   (meta m)                                           ;; => {:kw true :tag <class 'str'> :map :yes}

   ;; with-meta replaces the metadata on a copy
   (meta (with-meta m {:kw false}))                   ;; => {:kw false}

   ;; source object metadata remains unchanged
   (meta m)                                           ;; => {:kw true :tag <class 'str'> :map :yes}

.. seealso::

   :ref:`Reading metadata on literals <reader_metadata>`, :lpy:fn:`meta`, :lpy:fn:`with-meta`, :lpy:fn:`vary-meta`

.. _delays:

Delays
------

Delays are containers for deferring expensive computations until such time as the result is needed.
Create a new delay with the :lpy:fn:`delay` macro.
Results will not be computed until you attempt to :lpy:fn:`deref` or :lpy:fn:`force` evaluation.
Once a delay has been evaluated, it caches its results and returns the cached results on subsequent accesses.

.. code-block::

   (def d (delay (println "evaluating") (+ 1 2 3)))
   (force d)                                          ;; prints "evaluating"
                                                      ;; => 6
   (force d)                                          ;; does not print
                                                      ;; => 6

.. seealso::

   :lpy:fn:`delay`, :lpy:fn:`delay?`, :lpy:fn:`force`, :lpy:fn:`realized?`, :lpy:fn:`deref`

.. _promises:

Promises
--------

Promises are containers for receiving a deferred result, typically from another thread.
The value of a promise can be written exactly once using :lpy:fn:`deliver`.
Threads may await the results of the promise using a blocking :lpy:fn:`deref` call.

.. code-block::

   (def p (promise))
   (realized? p)                      ;; => false
   @(future (deliver p (+ 1 2 3)))
   (realized? p)                      ;; => true
   @p                                 ;; => 6
   (deliver p 7)                      ;; => nil
   @p                                 ;; => 6

.. seealso::

   :lpy:fn:`promise`, :lpy:fn:`deliver`, :lpy:fn:`realized?`, :lpy:fn:`deref`

.. _atoms:

Atoms
-----

Atoms are mutable, thread-safe reference containers which are useful for storing state that may need to be accessed (and changed) by multiple threads.
New atoms can be created with a default value using :lpy:fn:`atom`.
The state can be mutated in a thread-safe way using :lpy:fn:`swap!` and :lpy:fn:`reset!` (among others) without needing to coordinate with other threads.
Read the value of the atom using :lpy:fn:`deref`.

.. code-block::

   (def a (atom 0))
   (swap! a inc)       ;; => 1
   @a                  ;; => 1
   (swap! a #(+ 3 %))  ;; => 4
   @a                  ;; => 4
   (reset! a 0)        ;; => 0
   @a                  ;; => 0

Atoms are designed to contain one of Basilisp's immutable :ref:`data_structures`.
The ``swap!`` function in particular uses the :lpy:fn:`compare-and-set!` function to atomically swap in the results of applying the provided function to the existing value.
``swap!`` attempts to compare and set the value in a loop until it succeeds.
Since atoms may be accessed by multiple threads simultaneously, it is possible that the value of an atom has changed between when the state was polled and when the function finished computing its final result.
Update functions should therefore be free of side-effects since they may be called multiple times.

.. note::

   Atoms implement :py:class:`basilisp.lang.interfaces.IRef` and :py:class:`basilisp.lang.interfaces.IReference` and therefore support validators, watchers, and mutable metadata.

.. seealso::

   :lpy:fn:`atom`, :lpy:fn:`compare-and-set!`, :lpy:fn:`reset!`, :lpy:fn:`reset-vals!`, :lpy:fn:`swap!`, :lpy:fn:`swap-vals!`, :lpy:fn:`deref`, :ref:`reference_types`

.. _reference_types:

Reference Types
---------------

Basilisp's built-in reference types :ref:`vars` and :ref:`atoms` include support for metadata, validation, and watchers.

Unlike :ref:`metadata` on data structures, reference type metadata is mutable.
The identity of a reference type is the container, rather than the contained value, so it makes sense that if the value of a container can change so can the metadata.
:ref:`Var metadata <var_metadata>` is typically set at compile-time by a combination of compiler provided metadata and user metadata (typically via :lpy:form:`def`).
On the other hand, :ref:`atom <atoms>` have no metadata by default.
Metadata can be mutated using :lpy:fn:`alter-meta!` and :lpy:fn:`reset-meta!`.

Both Vars and atoms support validation of their contained value at the time it is set using a validator function.
Validator functions are functions of one argument returning either a single boolean value (where ``false`` indicates the value is invalid) or throwing an exception upon failure.
The validator will be called with the new proposed value of a ref before that value is applied.

.. code-block::

   (def a (atom 0))
   (set-validator! a (fn [v] (= 0 (mod v 2))))
   (swap! a inc)                                ;; => throws basilisp.lang.exception.ExceptionInfo: Invalid reference state {:data 1 :validator <...>}
   (swap! a #(+ 2 %))                           ;; => 2

Vars and atoms also feature support for watch functions which will be called on changes to the contained value.
Watch functions are functions of 4 arguments (watch key, reference value, old value, and new value).
Unlike validators, watches may not veto proposed changes to the contained value and any return value will be ignored.
A watch can be added to a reference using :lpy:fn:`add-watch` using a key and watches may be removed using :lpy:fn:`remove-watch` using the same key.

.. code-block::

   (def a (atom 0))
   (add-watch a :print (fn [_ r old new] (println r "changed from" old "to" new)))
   (swap! a inc)                 ;; => prints "<basilisp.lang.atom.Atom object at 0x113b01070> changed from 0 to 1"
                                 ;; => 1

.. note::

   Watch functions are called synchronously after a value change in an nondeterministic order.

.. warning::

   By the time a watch function is called, it is possible that the contained value has changed again, so users should use the provided arguments for the new and old value rather than attempting to :lpy:fn:`deref` the ref.

.. seealso::

   :ref:`atoms`, :ref:`vars`, :lpy:fn:`alter-meta!`, :lpy:fn:`reset-meta!`, :lpy:fn:`add-watch`, :lpy:fn:`remove-watch`, :lpy:fn:`get-validator`, :lpy:fn:`set-validator!`

.. _transients:

Transients
----------

Basilisp supports creating transient versions of most of its :ref:`persistent collections <data_structures>` using the :lpy:fn:`transient` function.
Transient versions of persistent data structures use local mutability to improve throughput for common data manipulation operations.
Because transients are mutable, they are intended to be used in local, single-threaded contexts where you may be constructing or modifying a collection.

Despite their mutability, the APIs for mutating transient collections are intentionally quite similar to that of standard persistent data structures.
Unlike classical data structure mutation APIs, you may not simply hang on to a single reference and issue repeated function calls or methods to that same data structure.
Instead, you use the transient-compatible variants of the existing persistent data structure functions (those ending with a ``!``) such as :lpy:fn:`assoc!`, :lpy:fn:`conj!`, etc.
As with the persistent data structures, you must use the return value from each of these functions as the input to subsequent operations.

Once you have completed modifying a transient, you should call :lpy:fn:`persistent!` to freeze the data structure back into its persistent variant.
After freezing a transient back into a persistent data structure, references to the transient are no longer guaranteed to be valid and may throw exceptions.

Many :lpy:ns:`basilisp.core` functions already use transients under the hood by default.
Take for example this definition of a function to merge an arbitrary number of maps (much like :lpy:fn:`merge`).

.. code-block::

   (defn merge [& maps]
     (when (some identity maps)
      (persistent!
       (reduce #(conj! %1 %2) (transient {}) maps))))

.. note::

   You can create transient versions of maps, sets, and vectors.
   Lists may not be made transient, since there would be no benefit.

.. warning::

   Transient data structures are not thread-safe and must therefore not be modified by multiple threads at once.
   It is the user's responsibility to ensure synchronization mutations to transients across threads.

.. seealso::

   :lpy:fn:`transient`, :lpy:fn:`persistent!`, :lpy:fn:`assoc!`, :lpy:fn:`conj!`, :lpy:fn:`disj!`, :lpy:fn:`dissoc!`, :lpy:fn:`pop!`

.. _volatiles:

Volatiles
---------

Volatiles are mutable, *non-thread-safe* reference containers which are useful for storing state that is mutable and is only changed in a single thread.
Create a new volatile using :lpy:fn:`volatile!`.
The stored value can be modified using :lpy:fn:`vswap!` and :lpy:fn:`vreset!`.

.. note::

   Volatiles are most frequently used for creating performant stateful :ref:`transducers`.

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

.. _data_types_and_records:

Data Types and Records
----------------------

TBD

.. seealso::

   :lpy:fn:`deftype`, :lpy:fn:`defrecord` , :lpy:fn:`record?`
