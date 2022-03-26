.. _special_forms:

Special Forms
=============

Special forms are the building blocks of any Lisp.
Special forms are fundamental forms which offer functionality directly from the base distribution.

.. lpy:currentns:: basilisp.core

.. _primary_special_forms:

Primary Special Forms
---------------------

.. lpy:specialform:: (def name)
                     (def name expr)
                     (def name docstring expr)

   Intern the value ``expr`` with the name ``name`` as a :ref:`Var <vars>` in the current namespace (the namespace pointed to by :lpy:var:`*ns*` in the current thread).

   ``name`` should be an unqualified :ref:`symbol <symbols>`.
   If a namespace is included with the symbol, it will be silently discarded by the compiler.
   :ref:`Metadata <metadata>` applied to the symbol ``name`` will be copied and applied to the interned Var.
   Common Var metadata applied via ``def`` include ``^:private``, ``^:dynamic``, and ``^:redef``.

   ``expr`` may be any valid expression.

   If no expression is expression is given, the Var is interned :ref:`unbound <unbound_vars>`.

   If a docstring is provided, the value of the docstring will be accessible on the ``:doc`` key of the Var meta.
   Docstrings must be :ref:`string literals <strings>`.
   References to names or Vars containing strings will be cause a compile-time error.

   .. code-block:: clojure

      (def my-var "Cool docstring!" :a-value)

      (:doc (meta (var my-var)))  ;;=> "Cool docstring!"
      (:doc (meta #'my-var))      ;;=> "Cool docstring!"

   .. note::

      By convention, ``def`` forms should only be used at the top level of a namespace file.
      While it is entirely legal to ``def`` a value within a function, the results of interning the Var within the function still apply to the current namespace.
      Within a function or method context, users should use the :lpy:form:`let` special form to bind a value to a name in that scope.

.. lpy:specialform:: (deftype name fields superclass+impls)

   Define a new data type (a Python class) with the given set of fields which implement 0 or more Python interfaces and Basilisp protocols.
   Types defined by ``deftype`` are immutable, slotted classes by default and do not include any niceties beyond what a basic Python class definition would give you.

   .. code-block:: clojure

      (defprotocol Shape
        (perimeter [self] "Return the perimeter of the Shape as a floating point number.")
        (area [self] "Return the area of the Shape as a floating point number."))

      (deftype Rectangle [x y]
        Shape
        (perimeter [self] (+ (* 2 x) (* 2 y)))
        (area [self] (* x y)))

   Fields should be given as a vector of names like a function argument list.
   Fields are accessible within implemented interface methods as unqualified names.
   Fields are immutable by default, but may be defined as mutable using the ``^:mutable`` metadata.
   Mutable fields may be set using the :lpy:form:`set!` special form from within any implemented interfaces.
   Fields may be given a default value using the ``{:default ...}`` metadata which will automatically be set when a new instance is created and which is not required to be provided during construction.
   Fields with defaults **must** appear after all fields without defaults.

   .. warning::

      Users should use field mutability and defaults sparingly, as it encourages exactly the types of design patterns that Basilisp and Clojure discourage.

   Python interfaces include any type which inherits from ``abc.ABC``\.
   New types may also implement all Python "dunder" methods automatically, though may also choose to explicitly "implement" ``python/object``.
   Python ``ABC`` types may include standard instance methods as well as class methods, properties, and static methods (unlike Java interfaces).
   Basilisp allows users to mark implemented methods as each using the ``^:classmethod``, ``^:property``, and ``^:staticmethod`` metadata, respectively, on the implemented method name.

   Neither the Python language specification nor the Python VM explicitly require users to use the ``abc.ABC`` metaclass and ``abc.abstractmethod`` decorator to define an abstract class or interface type, so a significant amount of standard library code and third-party libraries omit this step.
   As such, even if a class is functionally an abstract class or interface, the Basilisp compiler will not consider it one without ``abc.ABC`` in the superclass list.
   To get around this limitation, you can mark a class in the superclass list as "artificially" abstract using the ``^:abstract`` metadata.

   .. warning::

      Users should use artificial abstractness sparingly since it departs from the intended purpose of the ``deftype`` construct and circumvents protections built into the compiler.

   .. note::

      ``deftype`` is certainly necessary at times, but users should consider using :lpy:fn:`defrecord` first.
      ``defrecord`` creates a record type, which behaves like a map but which can also implement Python interfaces and satisfy Basilisp protocols.
      This makes it an ideal for data which needs to interact with Python code and Basilisp code.
      Records are strictly immutable, however, so they may not be suitable for all cases.

   .. seealso::

      :lpy:fn:`defrecord`, :lpy:fn:`defprotocol`, :lpy:form:`reify`

.. lpy:specialform:: (do)
                     (do & exprs)

   Wrap zero or more expressions in a block, returning the result of the last expression in the block.
   If no expressions are given, return ``nil``.

.. lpy:specialform:: (fn name? [& args] & body)
                     (fn name? ([args1 args2] & body) ([args1 args2 & rest] & body))

   Create a new anonymous function accepting zero or more arguments with zero or more body expressions.
   The result of calling the newly created function will be the final expression in the body, or ``nil`` if no body expressions are given.

   Anonymous functions may optionally be given a name which should be an unqualified :ref:`symbol <symbols>`.
   Function names may be useful in debugging as they will be used in stack traces.

   Function arguments should be :ref:`symbols` given in a :ref:`vector <vectors>`.
   Functions may be defined with zero or more arguments.
   For functions with a fixed number of positional arguments, it is a runtime error to call a function with the wrong number of arguments.
   Functions may accept a variadic number of arguments (called "rest" arguments by convention) by terminating their argument list with ``& rest``, with ``rest`` being any symbol name you choose.
   Rest arguments will be collected into a sequence which can be manipulated with the Basilisp sequence functions.

   .. note::

      Arguments in ``fn`` forms support :ref:`destructuring` which is an advanced tool for accessing specific portions of arguments.

   Functions may be overloaded with one or more arities (signature with different numbers of arguments).
   If a function has multiple arities, each arity should appear in its own :ref:`list <lists>` immediately after ``fn`` symbol or name if one is given.

   .. warning::

      All arities in a multi-arity function must have distinct numbers of arguments.
      It is a compile-time error to include two or more arities with the same number of arguments.

   .. warning::

      Multi-arity functions may only have zero or one arities which include a rest argument.
      It is a compile-time error to include multiple arities with rest arguments.

   .. warning::

      For multi-arity functions with a variadic arity, the variadic arity must have at least the same number of positional arguments as the maximum number of positional arguments across all of the remaining arities.
      It is a compile-time error to include a variadic arity in a multi-arity function with fewer fixed positional arguments than any other arity.

   .. note::

      Functions annotated with the ``:async`` metadata key will be compiled as Python coroutine functions (as by Python's `async def <https://docs.python.org/3/reference/compound_stmts.html#async-def>`_).
      Coroutine functions may make use of the :lpy:form:`await` special form.

.. lpy:specialform:: (if test true-expr)
                     (if test true-expr false-expr)

   Evaluate the expression ``test``, returning ``true-expr`` if ``test`` is truthy and ``false-expr`` otherwise.
   If no ``false-expr`` is given, it defaults to ``nil``.

   ``true-expr`` and ``false-expr`` may only be single expressions, so it may be necessary to combine ``if`` with :lpy:form:`do` for more complex conditionals.

   .. note::

      In Basilisp, only :ref:`nil` and :ref:`false <boolean_values>` are considered false by ``if`` -- all other expressions are truthy.
      This differs from Python, where many objects may be considered falsey if they are empty (such as lists, sets, and strings).

   .. seealso::

      :lpy:fn:`and`, :lpy:fn:`or`, :lpy:fn:`if-not`, :lpy:fn:`when`, :lpy:fn:`when-not`

.. lpy:specialform:: (. obj method)
                     (. obj method & args)
                     (. obj (method))
                     (. obj (method & args))
                     (.method obj)
                     (.method obj & args)

   Call the method ``method`` of ``obj`` with zero or more arguments.

   ``method`` must be an unqualified :ref:`symbol <symbols>`.

   .. note::

      Methods prefixed with a ``-`` will be treated as property accesses :lpy:form:`.-`, rather than method calls.

   .. seealso::

      :ref:`accessing_object_methods_and_props`

.. lpy:specialform:: (.- obj attr)
                     (.-attr obj)

   Access the attribute ``attr`` on object ``obj``.

   ``attr`` must be an unqualified :ref:`symbol <symbols>`.

   .. seealso::

      :ref:`accessing_object_methods_and_props`

.. lpy:specialform:: (let [& bindings] & body)

   Bind 0 or more symbol names to the result of expressions and execute the body of expressions with access to those expressions.
   Execute the body expressions in an implicit :lpy:form:`do`, returning the value of the final expression.
   As with ``do`` forms, if no expressions are given, returns ``nil``.

   Names bound in ``let`` forms are lexically scoped to the ``let`` body.
   Later binding expressions in ``let`` forms may reference the results of previously bound expressions.
   ``let`` form names may be rebound in child ``let`` and :lpy:form:`let` forms.

   .. note::

      Bindings in ``let`` forms support :ref:`destructuring` which is an advanced tool for accessing specific portions of arguments.

   .. code-block::

      (let [])  ;;=> nil

      (let [x 3]
        x)
      ;;=> 3

      (let [x 3
            y (inc x)]
        y)
      ;;=> 4

   .. note::

      Names bound in ``let`` forms are *not* variables and thus the value bound to a name cannot be changed.
      ``let`` form bindings may be overridden in child ``let`` and :lpy:form:`letfn` forms.

   .. note::

      Astute readers will note that the true "special form" is ``let*``, while :lpy:fn:`let` is a core macro which rewrites its inputs into ``let*`` forms.

.. lpy:specialform:: (letfn [& fns] & body)

   Bind 0 or more functions to names and execute the body of expressions with access to those expressions.
   Execute the body expressions in an implicit :lpy:form:`do`, returning the value of the final expression.
   As with ``do`` forms, if no expressions are given, returns ``nil``.

   Function names bound in ``letfn`` forms are lexically scoped to the ``letfn`` body.
   Functions in ``letfn`` forms may reference each other freely, allowing mutual recursion.
   ``letfn`` function names may be rebound in child :lpy:form:`let` and ``letfn`` forms.

   .. note::

      Function definitions in ``letfn`` forms support :ref:`destructuring` which is an advanced tool for accessing specific portions of arguments.

   .. code-block::

      (letfn [])  ;;=> nil

      (letfn [(plus-two [x] (+ (plus-one x) 1))
              (plus-one [x] (+ x 1))]
        (plus-two 3))
      ;;=> 4

   .. note::

      Names bound in ``letfn`` forms are *not* variables and thus the value bound to a name cannot be changed.
      ``letfn`` form bindings may be overridden in child :lpy:form:`let` and ``letfn`` forms.

   .. note::

      Astute readers will note that the true "special form" is ``letfn*``, while :lpy:fn:`letfn` is a core macro which rewrites its inputs into ``letfn*`` forms.

.. lpy:specialform:: (loop [& bindings] & body)

   ``loop`` forms are functionally identical to :lpy:form:`let` forms, save for the fact that ``loop`` forms establish a recursion point which enables looping with :lpy:form:`recur`.

   .. code-block::

      (loop [])  ;;=> nil

      (loop [x 3]
        x)
      ;;=> 3

      (loop [x 1]
        (if (< x 10)
          (recur (* x 2))
          x))
      ;;=> 16

   .. note::

      ``loop`` forms will not loop automatically -- users need to force the loop with :lpy:form:`recur`.
      Returning a value (rather than ``recur``\ing) from the loop terminates the loop and returns the final value.

   .. note::

      Astute readers will note that the true "special form" is ``loop*``, while :lpy:fn:`loop` is a core macro which rewrites its inputs into ``let*`` forms.

.. lpy:specialform:: (quote expr)

   Return the forms of ``expr`` unevaluated, rather than executing the expression.
   This is particularly useful in when writing macros.

   May also be shortened with the :ref:`special character <special_chars>` ``'``, as ``'form``.

   .. seealso::

      :ref:`macros`

.. lpy:specialform:: (recur & args)

   Evaluate the arguments given and re-binds them to the corresponding names at the last recursion point.
   Recursion points are defined for:

   * Each arity of a function created by :lpy:form:`fn` (and by extension :lpy:fn:`defn`).
     The number arguments to ``recur`` must match the arity of the recursion point.
     You may not recur between different arities of the same function.
   * Loops created via :lpy:form:`loop`\.
     The arguments to recur are rebound to the names in the ``loop`` binding.
   * Methods defined on types created via :lpy:form:`deftype`\.
     Users should not pass the ``self`` or ``this`` reference to ``recur``.
     ``recur`` is disallowed in static methods, class methods, and properties.

   .. note::

      All recursion with ``recur`` is tail-recursive by definition.
      It is a compile-time error to have a ``recur`` statement in non-tail position.

      Recursion points are checked lexically, so ``recur`` forms may only be defined in the same lexical context as a construct which defines a recursion point.

   .. note::

      Recursion via ``recur`` does not consume an additional stack frame in any case.
      Python does not support tail-call optimization, so users are discouraged from looping using traditional recursion for cases with unknown bounds.

.. lpy:specialform:: (reify superclass+impls)

   Return a new object which implements 0 or more Python interfaces and Basilisp protocols.
   Methods on objects returned by ``reify`` close over their environment, which provides a similar functionality to that of a class created by :lpy:form:`deftype`\.

   .. code-block:: clojure

      (defprotocol Shape
        (perimeter [self] "Return the perimeter of the Shape as a floating point number.")
        (area [self] "Return the area of the Shape as a floating point number."))

      (defn rectangle [x y]
        (reify Shape
          (perimeter [self] (+ (* 2 x) (* 2 y)))
          (area [self] (* x y))))

   Python interfaces include any type which inherits from ``abc.ABC``\.
   New types may also implement all Python "dunder" methods automatically, though may also choose to explicitly "implement" ``python/object``.
   Python ``ABC`` types may include standard instance methods as well as class methods, properties, and static methods (unlike Java interfaces).
   Basilisp allows users to mark implemented methods as each using the ``^:classmethod``, ``^:property``, and ``^:staticmethod`` metadata, respectively, on the implemented method name.

   Neither the Python language specification nor the Python VM explicitly require users to use the ``abc.ABC`` metaclass and ``abc.abstractmethod`` decorator to define an abstract class or interface type, so a significant amount of standard library code and third-party libraries omit this step.
   As such, even if a class is functionally an abstract class or interface, the Basilisp compiler will not consider it one without ``abc.ABC`` in the superclass list.
   To get around this limitation, you can mark a class in the superclass list as "artificially" abstract using the ``^:abstract`` metadata.

   .. warning::

      Users should use artificial abstractness sparingly since it departs from the intended purpose of the ``reify`` construct and circumvents protections built into the compiler.

   .. seealso::

      :lpy:form:`deftype`

.. lpy:specialform:: (set! target value)

   Set the ``target`` to the expression ``value``.
   Only a limited set of a targets are considered assignable:

   * :lpy:form:`deftype` locals designated as ``:mutable``
   * :ref:`Host fields <accessing_object_methods_and_props>`
   * :ref:`dynamic_vars` with established thread-local bindings

   .. note::

      The Basilisp compiler makes attempts to verify whether a ``set!`` is legal at compile time, but there are cases which must be deferred to runtime due to the dynamic nature of the language.
      In particular, due to the non-lexical nature of dynamic Var bindings, it can be difficult to establish if a Var is thread-bound when it is ``set!``, so this check is deferred to runtime.

.. lpy:specialform:: (throw exc)

   Throw the exception named by ``exc``.
   The semantics of ``throw`` are identical to those of Python's `raise <https://docs.python.org/3/reference/simple_stmts.html#the-raise-statement>`_ statement with exception.
   Unlike Python's ``raise``, an exception is always required and no explicit exception chaining is permitted (as by the ``from`` keyword in Python).

.. lpy:specialform:: (try *exprs *catch-exprs finally?)

   Execute 1 or more expressions (``exprs``) in an implicit :lpy:form:`do`, returning the final value if no exceptions occur.
   If an exception occurs and a matching ``catch`` expression is provided, handle the exception and return the value of the ``catch`` expression.
   Evaluation of which ``catch`` expression to use follows the semantics of the underlying Python VM -- that is, for an exception ``e``, bind to the first ``catch`` expression for which ``(instance? ExceptionType e)`` returns ``true``.
   Users may optionally provide a ``finally`` clause trailing the final ``catch`` expression which will be executed in all cases.

   .. note::

      Basilisp's ``try`` special form matches the semantics of Python's `try <https://docs.python.org/3/reference/compound_stmts.html#the-try-statement>`_ with two minor exceptions:

      * In Basilisp, a single ``catch`` expression may only bind to a single exception type.
      * In Basilisp, the ``finally`` clause can never provide a return value for the enclosing function.

.. lpy:specialform:: (var var-name)

   Access the :ref:`Var <vars>` named by ``var-name``.
   It is a compile-time exception if the Var cannot be resolved.

   May also be shortened to the :ref:`reader macro <reader_macros>` ``#'``.

   .. code-block:: clojure

      #'my-var

.. _basilisp_specific_special_forms:

Basilisp-specific Special Forms
-------------------------------

The special forms below were added to provide direct support for Python VM specific features and their usage should be relegated to platform-specific code.

.. lpy:specialform:: (await expr)

   Await a value from a function as by Python's `await <https://docs.python.org/3/reference/expressions.html#await-expression>`_ expression.
   Use of the ``await`` is only valid for functions defined as coroutine functions.
   See :lpy:form:`fn` for more information.

.. lpy:specialform:: (yield)
                     (yield expr)

   Yield a value from a function as by Python's `yield <https://docs.python.org/3/reference/simple_stmts.html#the-yield-statement>`_ statement.
   Use of the ``yield`` form automatically converts your function into a Python generator.
   Basilisp seq and sequence functions integrate seamlessly with Python generators.

.. _import_related_special_forms:

Import-related Special Forms
----------------------------

Basilisp provides two special forms specifically for importing Python and Basilisp code into the current context.

.. warning::

   These special forms should be considered an implementation detail and their direct usage is strongly discouraged.
   In nearly all cases, users should delegate to the corresponding functions in :lpy:ns:`basilisp.core` instead.

.. lpy:specialform:: (import* & py-packages)

   Import the Python package or packages given as arguments.
   See :lpy:fn:`import` for more details.

   .. warning::

      Basilisp namespaces should not be imported using this mechanism.
      See :lpy:form:`require` for more details on requiring Basilisp namespaces.

.. lpy:specialform:: (require* & namespaces)

   Load Basilisp libraries and make them accessible in the current namespace.
   See :lpy:fn:`require` for more details.

   .. warning::

      Python packages and modules cannot be imported using this mechanism.
      See :lpy:form:`import` for more details on importing Python modules.