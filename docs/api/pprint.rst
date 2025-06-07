basilisp.pprint
===============

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Pretty Printing
---------------

Pretty printing built-in data structures is as easy as a call to :lpy:fn:`pprint`.

.. code-block::

 (require '[basilisp.pprint :as pprint])
 (pprint/pprint (range 30))

The output can be configured using a number of different control variables, which
are expressed as dynamic Vars.

- :lpy:var:`*print-base*`
- :lpy:var:`*print-miser-width*`
- :lpy:var:`*print-pretty*`
- :lpy:var:`*print-pprint-dispatch*`
- :lpy:var:`*print-radix*`
- :lpy:var:`*print-right-margin*`
- :lpy:var:`*print-sort-keys*`
- :lpy:var:`*print-suppress-namespaces*`

You can pretty print the last result from the REPL using the :lpy:fn:`pp` convenience
macro.

As an alternative, the :lpy:fn:`write` API enables a more ergonomic API for
configuring the printer using keyword arguments rather than dynamic Vars.

.. code-block::

 (pprint/write (ns-interns 'basilisp.pprint) :sort-keys true)
 ;; {*current-length* #'basilisp.pprint/*current-length*
 ;;  ...
 ;;  write-out #'basilisp.pprint/write-out}

Custom Pretty Print Dispatch Function
-------------------------------------

TBD

Unimplemented Features
----------------------

The following features from ``clojure.pprint`` are not currently implemented:

- ``:fill`` newlines
- ``code-dispatch`` for printing code
- ``cl-format``

References
----------

- Tom Faulhaber et al.; ``clojure.pprint``
- Oppen, Derek; \"Prettyprinting\"; October 1980
- Waters, Richard; \"XP: A Common Lisp Pretty Printing System\"; March 1989

API
---

.. autonamespace:: basilisp.pprint
   :members:
   :undoc-members:
   :exclude-members: LogicalBlock, StartBlock, EndBlock, Blob, Newline, Indent, *current-length*, *current-level*