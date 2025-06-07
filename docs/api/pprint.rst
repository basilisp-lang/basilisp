basilisp.pprint
===============

.. lpy:currentns:: basilisp.pprint

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. _pretty_printing:

Pretty Printing
---------------

Pretty printing built-in data structures is as easy as a call to :lpy:fn:`pprint`.

.. code-block::

 (require '[basilisp.pprint :as pprint])
 (pprint/pprint (range 30))

The output can be configured using a number of different control variables, which are expressed as dynamic Vars.

- :lpy:var:`*print-base*`
- :lpy:var:`*print-miser-width*`
- :lpy:var:`*print-pretty*`
- :lpy:var:`*print-pprint-dispatch*`
- :lpy:var:`*print-radix*`
- :lpy:var:`*print-right-margin*`
- :lpy:var:`*print-sort-keys*`
- :lpy:var:`*print-suppress-namespaces*`

You can pretty print the last result from the REPL using the :lpy:fn:`pp` convenience macro.

As an alternative, the :lpy:fn:`write` API enables a more ergonomic API for configuring the printer using keyword arguments rather than dynamic Vars.

.. code-block::

 (pprint/write (ns-interns 'basilisp.pprint) :sort-keys true)
 ;; {*current-length* #'basilisp.pprint/*current-length*
 ;;  ...
 ;;  write-out #'basilisp.pprint/write-out}

.. _custom_pretty_print_dispatch_function:

Custom Pretty Print Dispatch Function
-------------------------------------

The default dispatch function is :lpy:fn:`simple-dispatch` which can print most builtin Basilisp types.
Using the builtin macros and utilities, it is possible to create a custom dispatch function.

.. _pretty_printing_concepts:

Pretty Printing Concepts
^^^^^^^^^^^^^^^^^^^^^^^^

The pretty printing algorithm used in ``basilisp.pprint`` is based on the XP algorithm defined in Richard Water's 1989 paper "XP: A Common Lisp Pretty Printing System" as adapted in Clojure's ``pprint`` by Tom Faulhaber.
There are three basic concepts in the XP algorithm which are necessary in order to create a custom dispatch function.

- *Logical blocks* are groups of output that should be treated as a single unit by the pretty printer.
  Logical blocks can nest, so one logical block may contain 0 or more other logical blocks.
  For example, a vector may contain a map; the vector would be a logical block and the map would also be a logical block.
  ``simple-dispatch`` even treats key/value pairs in associative type outputs as a logical block, so they are printed on the same line whenever possible.

  A dispatch function can emit a logical block using the :lpy:fn:`pprint-logical-block` macro.

- *Conditional newlines* can be emitted anywhere a newline may need inserted into the output stream.
  Newlines can be one of 3 different types which hints to the pretty printer when a newline should be emitted.

  Dispatch functions can emit newlines in any supported style using the :lpy:fn:`pprint-newline` function.

  - ``:linear`` style newlines should be emitted whenever the enclosing logical block does not fit on a single line.
    Note that if any linear newline is emitted in a block, every linear newline will be emitted in that block.

  - ``:mandatory`` style newlines are emitted in all cases.

  - ``:miser`` style newlines are emitted only when the output will occur in the "miser" region as defined by :lpy:var:`*print-miser-width*`.
    This allows additional newlines to be emitted as the output nests closer to the right margin.

- *Indentation* commands indicate how indentation of subsequent lines in a logical block should be defined.
  Indentation may be defined relative to either the starting column of the current logical block or to the current column of the output.

  Dispatch functions can control indentation using the :lpy:fn:`pprint-indent` function.

Pretty printing is most useful for viewing large, nested structures in a more human-friendly way.
To that end, dispatch functions wishing to print any collection may want to use the :lpy:fn:`print-length-loop` macro to loop over the output, respecting the :lpy:var:`basilisp.core/*print-length*` setting.

Dispatch functions which may need to be called on nested elements should use :lpy:fn:`write-out` to ensure that :lpy:var:`basilisp.core/*print-level*` is respected.
Scalar values can be printed with :lpy:fn:`basilisp.core/pr` or just written directly to :lpy:var:`*out*`.

.. _unimplemented_pprint_features:

Unimplemented Features
----------------------

The following features from ``clojure.pprint`` are not currently implemented:

- ``:fill`` newlines
- ``code-dispatch`` for printing code
- ``cl-format``

.. _pprint_references:

References
----------

- Tom Faulhaber et al.; ``clojure.pprint`` (`API <https://clojure.github.io/clojure/clojure.pprint-api.html>`_, `Documentation <https://clojure.github.io/clojure/doc/clojure/pprint/PrettyPrinting.html>`_)
- Oppen, Derek; \"Prettyprinting\"; October 1980
- Waters, Richard; \"XP: A Common Lisp Pretty Printing System\"; March 1989

.. _pprint_api:

API
---

.. autonamespace:: basilisp.pprint
   :members:
   :undoc-members:
   :exclude-members: LogicalBlock, StartBlock, EndBlock, Blob, Newline, Indent, *current-length*, *current-level*