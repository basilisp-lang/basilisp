.. _special_forms:

Special Forms
=============

Special forms are the building blocks of any Lisp.
Special forms are fundamental forms which offer functionality directly from the base distribution.

.. lpy:currentns:: basilisp.core

.. lpy:specialform:: (await expr)

   TBD

.. lpy:specialform:: (def name)
                     (def name expr)
                     (def name docstring expr)

   Intern the value ``expr`` with the name ``name`` as a :ref:`Var <vars>` in the current namespace (the namespace pointed to by :lpy:var:`*current-ns*` in the current thread).

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

.. lpy:specialform:: (deftype ...)

   TBD

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

.. lpy:specialform:: (import & py-packages)

   Import the Python package or packages given as arguments.
   Package names should be unqualified :ref:`symbols` or three element :ref:`vectors`.
   The vector form is ``[package-name :as alias]`` and the name will be bound as the chosen ``alias`` rather than as the package name.
   Package symbols in both the symbol and vector format may include dots which will behave in the expected way, consistent with standard Python ``import`` statements.

   .. warning::

      Basilisp namespaces should not be imported using this mechanism.
      While it may work for basic use cases, it may introduce unexpected and hard-to-diagnose bugs.
      Instead, Basilisp namespaces should be imported using :lpy:form:`require`.

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

.. lpy:specialform:: (let [& args] & body)

   Bind 0 or more symbol names to the result of expressions and execute the body of expressions with access to those expressions.

.. lpy:specialform:: (letfn ...)

   TBD

.. lpy:specialform:: (loop ...)

   TBD

.. lpy:specialform:: (quote expr)

   Return the forms of ``expr`` as data, rather than executing the expression.
   This is particularly useful in when writing macros.

   .. seealso::

      :ref:`macros`

.. lpy:specialform:: (recur ...)

   TBD

.. lpy:specialform:: (reify ...)

   TBD

.. lpy:specialform:: (require ...)

   TBD

.. lpy:specialform:: (set! ...)

   TBD

.. lpy:specialform:: (throw ...)

   TBD

.. lpy:specialform:: (try ...)

   TBD

.. lpy:specialform:: (var var-name)

   Access the :ref:`Var <vars>` named by ``var-name``.
   It is a compile-time exception if the Var cannot be resolved.

   May also be shortened to the :ref:`reader macro <reader_macros>` ``#'``.

   .. code-block:: clojure

      #'my-var

.. lpy:specialform:: (yield ...)

   TBD