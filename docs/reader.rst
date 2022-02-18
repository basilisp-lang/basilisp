Reader
======

In most Lisps, the reader is the component responsible for reading in the textual representation of the program into memory as data structures.
Lisps are typically referred to as *homoiconic*, since that representation typically matches the syntax tree exactly in memory.
This is in contrast to a non-homoiconic language such as Java or Python, which typically parses a textual program into an abstract syntax tree which captures the *meaning* of the textual program, but not necessarily the structure.

The Basilisp reader performs a job which is a combination of the traditional lexer and parser.
The reader takes a file or string and produces a stream of Basilisp data structures.
Typically the reader streams its results to the compiler, but end-users may also take advantage of the reader directly from within Basilisp.

.. contents:: Reader Literals
   :depth: 2

.. _numeric_literals:

Numeric Literals
----------------

The Basilisp reader reads a wide range of numeric literals.

Integers
^^^^^^^^

::

    basilisp.user=> 1
    1
    basilisp.user=> (python/type 1)
    <class 'int'>
    basilisp.user=> 1N
    1
    basilisp.user=> (python/type 1N)
    <class 'int'>

Integers are represented using numeric ``0-9`` and may be prefixed with any number of negative signs ``-``.
The resulting integer will have the correct sign after resolving all of the supplied ``-`` signs.
For interoperability support with Clojure, Basilisp integers may also be declared with the ``N`` suffix, like ``1N``.
In Clojure, this syntax signals a ``BigInteger``, but Python's default ``int`` type supports arbitrary precision by default so there is no difference between ``1`` and ``1N`` in Basilisp.

Floating Point
^^^^^^^^^^^^^^

::

   basilisp.user=> 1.0
   1.0
   basilisp.user=> (python/type 1.0)
   <class 'float'>
   basilisp.user=> 1M
   1
   basilisp.user=> (python/type 1M)
   <class 'decimal.Decimal'>

Floating point values are represented using ``0-9`` and a trailing decimal value, separated by a ``.`` character.
Like integers, floating point values may be prefixed with an arbitrary number of negative signs ``-`` and the final read value will have the correct sign after resolving the negations.
By default floating point values are represented by Python's ``float`` type, which does **not** support arbitrary precision by default.
Like in Clojure, floating point literals may be specified with a single ``M`` suffix to specify an arbitrary-precision floating point value.
In Basilisp, a floating point number declared with a trailing ``M`` will return Python's `Decimal <https://docs.python.org/3/library/decimal.html>`_ type, which supports arbitrary floating point arithmetic.

Complex
^^^^^^^

::

    basilisp.user=> 1J
    1J
    basilisp.user=> (python/type 1J)
    <class 'complex'>
    basilisp.user=> 1.0J
    1J
    basilisp.user=> (python/type 1.0J)
    <class 'complex'>

Basilisp includes support for complex literals to match the Python VM hosts it.
Complex literals may be specified as integer or floating point values with a ``J`` suffix.
Like integers and floats, complex values may be prefixed with an arbitrary number of negative signs ``-`` and the final read value will have the correct sign after resolving the negations.

.. _strings:

Strings
-------

::

    basilisp.user=> ""
    ""
    basilisp.user=> "this is a string"
    "this is a string"
    basilisp.user=> (python/type "")
    <class 'str'>

Strings are denoted as a series of characters enclosed by ``"`` quotation marks.
If a string needs to contain a quotation mark literal, that quotation mark should be escaped as ``\"``.
Strings may be multi-line by default and only a closing ``"`` will terminate reading a string.
Strings correspond to the Python ``str`` type.

.. _character_literals:

Character Literals
------------------

::

    basilisp.user=> \a
    "a"
    basilisp.user=> \u03A9
    "Ω"
    basilisp.user=> \newline
    "
    "

For Clojure compatibility, character literals may be specified in code prefixed by a ``\`` character.
Character literals are actually backed by Python strings, as Python does not have a true *character* type.

The reader supports 6 special character literal names for common whitespace characters: ``\newline``, ``\space``, ``\tab``, ``\formfeed``, ``\backspace``, ``\return``.

Unicode code points may be specified as ``\uXXXX`` where ``XXXX`` corresponds to the hex-code for unicode code point.

Otherwise, characters may be specified as ``\a``, which will simply yield the character as a string.

.. _boolean_values:

Boolean Values
--------------

::

    basilisp.user=> true
    true
    basilisp.user=> (python/type true)
    <class 'bool'>
    basilisp.user=> false
    false
    basilisp.user=> (python/type false)
    <class 'bool'>

The special values ``true`` and ``false`` correspond to Python's ``True`` and ``False`` respectively.

.. _nil:

nil
---

::

    basilisp.user=> nil
    nil
    basilisp.user=> (python/type nil)
    <class 'NoneType'>

The special value ``nil`` correspond's to Python's ``None``.

.. _whitespace:

Whitespace
----------

Characters typically considered as whitespace are also considered whitespace by the reader and ignored.
Additionally, the ``,`` character is considered whitespace and will be ignored.
This allows users to optionally comma-separate collection-literal elements and key-value pairs in map literals.

.. _symbols:

Symbols
-------

::

    basilisp.user=> 'sym
    sym
    basilisp.user=> 'namespaced/sym
    namespaced/sym

Symbolic identifiers, most often used to refer to a Var or value in Basilisp.
Symbols may optionally include a namespace, which is delineated from the *name* of the symbol by a ``/`` character.

Symbols may be represented with most word characters and some punctuation marks which are typically reserved in other languages, such as: ``-``, ``+``, ``*``, ``?``, ``=``, ``!``, ``&``, ``%``, ``>``, and ``<``.

.. _keywords:

Keywords
--------

::

    basilisp.user=> :keyword
    :keyword
    basilisp.user=> :namespaced/keyword
    :namespaced/keyword

Keywords are denoted by the ``:`` prefix character.
Keywords can be viewed as a mix between :ref:`strings` and :ref:`symbols` in that they are often used as symbolic identifiers, but more typically for data rather than for code.
Like Symbols, keywords can contain an optional namespace, also delineated from the *name* of the keyword by a ``/`` character.

Keywords may be represented with most word characters and some punctuation marks which are typically reserved in other languages, such as: ``-``, ``+``, ``*``, ``?``, ``=``, ``!``, ``&``, ``%``, ``>``, and ``<``.

Keyword values are interned and keywords are compared by identity, not by value.

.. _lists:

Lists
-----

::

    basilisp.user=> ()
    ()
    basilisp.user=> '(1 "2" :three)
    (1 "2" :three)

Lists are denoted with the ``()`` characters.
Lists may contain 0 or more other heterogeneous elements.
Basilisp lists are classical Lisp singly-linked lists.
Non-empty list literals are not required to be prefixed by the quote ``'`` character for the reader, but they are shown quoted since the REPL also compiles the expression.

.. _vectors:

Vectors
-------

::

    basilisp.user=> []
    []
    basilisp.user=> [1 "2" :three]
    [1 "2" :three]

Vectors are denoted with the ``[]`` characters.
Vectors may contain 0 or more other heterogeneous elements.
Basilisp vectors are modeled after Clojure's persistent vector implementation.

.. _maps:

Maps
----

::

    basilisp.user=> {}
    {}
    basilisp.user=> {1 "2" :three 3}
    {1 "2" :three 3}

Maps are denoted with the ``{}`` characters.
Sets may contain 0 or more heterogenous key-value pairs.
Basilisp maps are modeled after Clojure's persistent map implementation.

.. _sets:

Sets
----

::

    basilisp.user=> #{}
    #{}
    basilisp.user=> #{1 "2" :three}
    #{1 "2" :three}

Sets are denoted with the ``#{}`` characters.
Sets may contain 0 or more other heterogeneous elements.
Basilisp sets are modeled after Clojure's persistent set implementation.

.. _line_comments:

Line Comments
-------------

Line comments are specified with the ``;`` character.
All of the text to the end of the line are ignored.

For a convenience in writing shell scripts with Basilisp, the standard \*NIX `shebang <https://en.wikipedia.org/wiki/Shebang_(Unix)>` (``#!``) is also treated as a single-line comment.

.. _metadata:

Metadata
--------

::

    basilisp.user=> (meta '^:macro s)
    {:macro true}
    basilisp.user=> (meta '^str s)
    {:tag str}
    basilisp.user=> (meta '^{:has-meta true} s)
    {:has-meta true}

Metadata can be applied to the following form by specifying metadata before the form as ``^meta form``.

The following builtin types support metadata: :ref:`symbols`, :ref:`lists`, :ref:`vectors`, :ref:`maps`, and :ref:`sets`.

Metadata applied to a form must be one of: :ref:`maps`, :ref:`symbols`, :ref:`keywords`:

* Symbol metadata will be normalized to a Map with the symbol as the value for the key ``:tag``.
* Keyword metadata will be normalized to a Map with the keyword as the key with the value of ``true``.
* Map metadata will not be modified when it is read.

.. _reader_macros:

Reader Macros
-------------

Basilisp supports most of the same reader macros as Clojure.
Reader macros are always dispatched using the ``#`` character.

* ``#'form`` is rewritten as ``(var form)``.
* ``#_form`` causes the reader to completely ignore ``form``.
* ``#!form`` is treated as a single-line comment (like ``;form``) as a convenience to support `shebangs <https://en.wikipedia.org/wiki/Shebang_(Unix)>` at the top of Basilisp scripts.
* ``#"str"`` causes the reader to interpret ``"str"`` as a regex and return a Python `re.pattern <https://docs.python.org/3/library/re.html>`_.
* ``#(...)`` causes the reader to interpret the contents of the list as an anonymous function. Anonymous functions specified in this way can name arguments using ``%1``, ``%2``, etc. and rest args as ``%&``. For anonymous functions with only one argument, ``%`` can be used in place of ``%1``.

.. _data_readers:

Data Readers
------------

Data readers are reader macros which can take in un-evaluated forms and return new forms.
This construct allows end-users to customize the reader to read otherwise unsupported custom literal syntax for commonly used data.

Data readers are specified with the ``#`` dispatch prefix, like reader macros, and are followed by a symbol.
User-specified data reader symbols must include a namespace, but builtin data readers are not namespaced.

Basilisp supports a few builtin data readers:

* ``#inst "2018-09-14T15:11:20.253-00:00"`` yields a Python `datetime <https://docs.python.org/3/library/datetime.html#datetime-objects>`_ object.
* ``#uuid "c3598794-20b4-48db-b76e-242f4405743f"`` yields a Python `UUID <https://docs.python.org/3/library/uuid.html#uuid.UUID>`_ object.

One of the benefits of choosing Basilisp is convenient built-in Python language interop.
However, the immutable data structures of Basilisp may not always play nicely with code written for (and expecting to be used by) other Python code.
Fortunately, Basilisp includes data readers for reading Python collection literals directly from the REPL or from Basilisp source.

Python literals can be read by prefixing existing Basilisp data structures with a ``#py`` data reader tag.
Python literals use the matching syntax to the corresponding Python data type, which does not always match the syntax for the same data type in Basilisp.

* ``#py []`` produces a Python `list <https://docs.python.org/3/library/stdtypes.html#list>` type.
* ``#py ()`` produces a Python `tuple <https://docs.python.org/3/library/stdtypes.html#tuple>` type.
* ``#py {}`` produces a Python `dict <https://docs.python.org/3/library/stdtypes.html#dict>` type.
* ``#py #{}`` produces a Python `set <https://docs.python.org/3/library/stdtypes.html#set>` type.

.. _special_chars:

Special Characters
------------------

Basilisp's reader has a few special characters which cause the reader to emit modified forms:

* ``'form`` is rewritten as ``(quote form)``.
* ``@form`` is rewritten to ``(basilisp.core/deref form)``.

.. _syntax_quoting:

Syntax Quoting
--------------

Syntax quoting is a facility primarily used for writing macros in Basilisp.

.. _reader_conditions:

Reader Conditionals
-------------------

Reader conditionals are a powerful reader feature which allow Basilisp to read code written for other Clojure-like platforms (such as Clojure JVM or ClojureScript) without experiencing catastrophic errors.
Platform-specific Clojure code can be wrapped in reader conditionals and the reader will match only forms identified by supported reader "features".
Features are just standard :ref:`keywords`.
By default, Basilisp supports the ``:lpy`` feature.

Reader conditionals appear as Basilisp lists prefixed with the ``#?`` characters.
Like maps, reader conditionals should always contain an even number of forms.
Each pair should consist of the keyword used to identify the platform feature (such as ``:lpy`` for Basilisp) and the intended form for that feature.
The reader may emit no forms (much like with the :ref:`reader_macros` ``#_``) if there are no supported features in the reader conditional form.

::

    basilisp.user=> #?(:clj 1 :lpy 2)
    2
    basilisp.user=> #?(:clj 1)
    basilisp.user=>
    basilisp.user=> [#?@(:lpy [1 2 3])]
    [1 2 3]

For advanced use cases, reader conditionals may also be written to splice their contents into surrounding forms.
Splicing reader conditionals are subject to the same rules as splicing unquote in a syntax quoting context.
Splicing reader conditionals may only appear within other collection literal forms (such as lists, maps, sets, and vectors).

::

    basilisp.user=> [#?@(:lpy [1 2 3])]
    [1 2 3]
    basilisp.user=> #?@(:lpy [1 2 3])
    basilisp.lang.reader.SyntaxError: Unexpected reader conditional
