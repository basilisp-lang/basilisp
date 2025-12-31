.. _reader:

Reader
======

In most Lisps, the reader is the component responsible for reading in the textual representation of the program into memory as data structures.
Lisps are typically referred to as *homoiconic*, since that representation typically matches the syntax tree exactly in memory.
This is in contrast to a non-homoiconic language such as Java or Python, which typically parses a textual program into an abstract syntax tree which captures the *meaning* of the textual program, but not necessarily the structure.

The Basilisp reader performs a job which is a combination of the traditional lexer and parser.
The reader takes a file or string and produces a stream of Basilisp data structures.
Typically the reader streams its results to the compiler, but end-users may also take advantage of the reader directly from within Basilisp.

.. _reader_numeric_literals:

Numeric Literals
----------------

The Basilisp reader reads a wide range of numeric literals.

.. seealso::

   :ref:`numbers`

.. _reader_integer_numbers:

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

Integers are represented using numeric ``0-9`` and may be prefixed with a single negative sign ``-``.
For interoperability support with Clojure, Basilisp integers may also be declared with the ``N`` suffix, like ``1N``.
In Clojure, this syntax signals a ``BigInteger``, but Python's default ``int`` type supports arbitrary precision by default so there is no difference between ``1`` and ``1N`` in Basilisp.

Integer literals may be specified in arbitrary bases between 2 and 36 by using the syntax ``[base]r[value]``.
For example, in base 2 ``2r1001``, base 12 ``12r918a32``, and base 36 ``36r81jdk3kdp``.
Arbitrary base literals do not distinguish between upper and lower case characters, so ``p`` and ``P`` are the same for bases which support ``P`` as a digit.
Arbitrary base literals do not support the ``N`` suffix because ``N`` is a valid digit for some bases.

For common bases such as octal and hex, there is a custom syntax.
Octal literals can be specified with a ``0`` prefix; for example, the octal literal ``0777`` corresponds to the base 10 integer 511.
Hex literals can be specified with a ``0x`` prefix; for example, the hex literal ``0xFACE`` corresponds to the base 10 integer 64206.
Both octal and hex literals support the ``N`` suffix and it is treated the same as with base 10 integers.

.. _reader_floating_point_numbers:

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
Like integers, floating point values may be prefixed with a single negative sign ``-``.
By default floating point values are represented by Python's ``float`` type, which does **not** support arbitrary precision by default.
Like in Clojure, floating point literals may be specified with a single ``M`` suffix to specify an arbitrary-precision floating point value.
In Basilisp, a floating point number declared with a trailing ``M`` will return Python's :external:py:class:`decimal.Decimal` type, which supports user specified precision.

.. _reader_scientific_notation:

Scientific Notation
^^^^^^^^^^^^^^^^^^^

::

   basilisp.user=> 2e6
   2000000
   basilisp.user=> 3.14e-1
   0.31400000000000006

Basilisp supports scientific notation using the ``e`` syntax common to many programming languages.
The significand (the number to the left of the ``e`` ) may be an integer or floating point and may be prefixed with a single negative sign ``-`` or plus sign ``+``.
The exponent (the number to the right of the ``e`` ) must be an integer and may be prefixed with a single negative sign ``-``.
The resulting value will be either an integer or float depending on the type of the significand.

.. _reader_complex_numbers:

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
Like integers and floats, complex values may be prefixed with a single negative sign ``-``.

.. _reader_ratios:

Ratios
^^^^^^

::

   basilisp.user=> 22/7
   22/7
   basilisp.user=> -3/8
   -3/8

Basilisp includes support for ratios.
Ratios are represented as the division of 2 integers which cannot be reduced to an integer.
As with integers and floats, the numerator of a ratio may be prefixed with a single negative sign ``-`` -- a negative sign may not appear in the denominator.
In Basilisp, ratios are backed by Python's :external:py:class:`fractions.Fraction` type, which is highly interoperable with other Python numeric types.

.. _reader_strings:

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
String literals are always read with the UTF-8 encoding.

String literals may contain the following escape sequences: ``\\``, ``\a``, ``\b``, ``\f``, ``\n``, ``\r``, ``\t``, ``\v``.
Their meanings match the equivalent escape sequences supported in `Python string literals <https://docs.python.org/3/reference/lexical_analysis.html#string-and-bytes-literals>`_\.

String literals may also contain Unicode escape sequences prefixed with ``\u`` or ``\U``.
The escape sequence must be followed by either exactly 4 or exactly 8 hexadecimal characters, which denote a Unicode character with the given hex value.
Unlike in Python, you may prefix with either ``\u`` or ``\U`` without respect to the number of hex digits which follow.

.. seealso::

   :ref:`strings_and_byte_strings`

.. _reader_f_strings:

f-strings
^^^^^^^^^

::

    basilisp.user=> #f ""
    ""
    basilisp.user=> (let [a 1] #f "this is a string with {(inc a)}")
    "this is a string with 2"
    basilisp.user=> (let [a 1] #f "this is a string with \{(inc a)}")
    "this is a string with {(inc a)}"

f-strings are denoted as a series of characters enclosed by ``"`` quotation marks and preceded by a ``#f``.
Expressions may be interpolated in the string enclosed in ``{}``.
Each interpolation must contain exactly 1 expression and may be surrounded by optional whitespace characters which will not be included in the final string.
Any valid expression may appear in a string interpolation, including another string.
To include a literal opening ``{`` character, it must be escaped as ``\{``.

f-strings are otherwise identical to standard :ref:`string literals <reader_strings>`.

.. _reader_byte_strings:

Byte Strings
------------

::

    basilisp.user=> #b ""
    #b ""
    basilisp.user=> #b "this is a string"
    #b "this is a string"
    basilisp.user=> (type #b "")
    <class 'bytes'>

Byte strings are denoted as a series of ASCII characters enclosed by ``"`` quotation marks and preceded by a ``#b``.
If a string needs to contain a quotation mark literal, that quotation mark should be escaped as ``\"``.
Strings may be multi-line by default and only a closing ``"`` will terminate reading a string.
Strings correspond to the Python ``bytes`` type.

Byte string literals may contain the following escape sequences: ``\\``, ``\a``, ``\b``, ``\f``, ``\n``, ``\r``, ``\t``, ``\v``.
Byte strings may also characters using a hex escape code as ``\xhh`` where ``hh`` is a hexadecimal value.
Their meanings match the equivalent escape sequences supported in `Python byte string literals <https://docs.python.org/3/reference/lexical_analysis.html#string-and-bytes-literals>`_\.

.. warning::

   As in Python, byte string literals may not include any characters outside of the ASCII range.

.. seealso::

   :ref:`strings_and_byte_strings`

.. _reader_character_literals:

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

.. _reader_boolean_values:

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

.. seealso::

   :ref:`boolean_values`

.. _reader_nil:

nil
---

::

    basilisp.user=> nil
    nil
    basilisp.user=> (python/type nil)
    <class 'NoneType'>

The special value ``nil`` corresponds to Python's ``None``.

.. seealso::

   :ref:`nil`

.. _reader_whitespace:

Whitespace
----------

Characters typically considered as whitespace are also considered whitespace by the reader and ignored.
Additionally, the ``,`` character is considered whitespace and will be ignored.
This allows users to optionally comma-separate collection-literal elements and key-value pairs in map literals.

.. _reader_symbols:

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

.. seealso::

   :ref:`symbols`

.. _reader_keywords:

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

Keywords may be represented with most word characters and some punctuation marks which are typically reserved in other languages, such as: ``.``, ``-``, ``+``, ``*``, ``?``, ``=``, ``!``, ``&``, ``%``, ``>``, and ``<``.

Keywords prefixed with a double colon ``::`` are will have their namespace automatically resolved to the current namespace or, if an alias is specified, to the full name associated with the given alias in the current namespace.

.. seealso::

   :ref:`keywords`

.. _reader_lists:

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

.. seealso::

   :ref:`lists`

.. _reader_vectors:

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

.. seealso::

   :ref:`vectors`

.. _reader_maps:

Maps
----

::

    basilisp.user=> {}
    {}
    basilisp.user=> {1 "2" :three 3}
    {1 "2" :three 3}

Maps are denoted with the ``{}`` characters.
Maps may contain 0 or more heterogenous key-value pairs.
Basilisp maps are modeled after Clojure's persistent map implementation.

.. seealso::

   :ref:`maps`

.. _reader_sets:

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

.. seealso::

   :ref:`sets`

.. _reader_line_comments:

Line Comments
-------------

Line comments are specified with the ``;`` character.
All of the text to the end of the line are ignored.

For a convenience in writing shell scripts with Basilisp, the standard \*NIX `shebang <https://en.wikipedia.org/wiki/Shebang_(Unix)>`_ (``#!``) is also treated as a single-line comment.

.. _reader_metadata:

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
* Vector metadata will be normalized to a Map with the vector as the value for the key ``:param-tags``.
* Map metadata will not be modified when it is read.

.. seealso::

   :ref:`metadata`

.. _reader_macros:

Reader Macros
-------------

Basilisp supports most of the same reader macros as Clojure.
Reader macros are always dispatched using the ``#`` character.

* ``#'form`` is rewritten as ``(var form)``.
* ``#_form`` causes the reader to completely ignore ``form``.
* ``#!form`` is treated as a single-line comment (like ``;form``) as a convenience to support `shebangs <https://en.wikipedia.org/wiki/Shebang_(Unix)>`_ at the top of Basilisp scripts.
* ``#"str"`` causes the reader to interpret ``"str"`` as a regex and return a Python :external:py:mod:`re.pattern <re>`.
* ``#(...)`` causes the reader to interpret the contents of the list as an anonymous function. Anonymous functions specified in this way can name arguments using ``%1``, ``%2``, etc. and rest args as ``%&``. For anonymous functions with only one argument, ``%`` can be used in place of ``%1``.

.. _data_readers:

Data Readers
------------

Data readers are reader macros which can take in un-evaluated forms and return new forms.
This construct allows end-users to customize the reader to read otherwise unsupported custom literal syntax for commonly used data.

Data readers are specified with the ``#`` dispatch prefix, like reader macros, and are followed by a symbol.
User-specified data reader symbols must include a namespace, but builtin data readers are not namespaced.

Basilisp supports a few builtin data readers:

* ``#inst "2018-09-14T15:11:20.253-00:00"`` yields a Python :external:py:class:`datetime.datetime` object.
* ``#uuid "c3598794-20b4-48db-b76e-242f4405743f"`` yields a Python :external:py:class`uuid.UUID` object.

One of the benefits of choosing Basilisp is convenient built-in Python language interop.
However, the immutable data structures of Basilisp may not always play nicely with code written for (and expecting to be used by) other Python code.
Fortunately, Basilisp includes data readers for reading Python collection literals directly from the REPL or from Basilisp source.

Python literals can be read by prefixing existing Basilisp data structures with a ``#py`` data reader tag.
Python literals use the matching syntax to the corresponding Python data type, which does not always match the syntax for the same data type in Basilisp.

* ``#py []`` produces a Python `list <https://docs.python.org/3/library/stdtypes.html#list>`_ type.
* ``#py ()`` produces a Python `tuple <https://docs.python.org/3/library/stdtypes.html#tuple>`_ type.
* ``#py {}`` produces a Python `dict <https://docs.python.org/3/library/stdtypes.html#dict>`_ type.
* ``#py #{}`` produces a Python `set <https://docs.python.org/3/library/stdtypes.html#set>`_ type.

.. _custom_data_readers:

Custom Data Readers
^^^^^^^^^^^^^^^^^^^

`Like Clojure <https://clojure.org/reference/reader#tagged_literals>`_ , data readers can be changed by binding  :lpy:var:`*data-readers*`.

When Basilisp starts it can load data readers from multiple sources.

It will search in the top level directory and up to its immediate subdirectories (which typically representing installed modules) of the :external:py:data:`sys.path` entries for files named ``data_readers.lpy`` or else ``data_readers.cljc``; each which must contain a mapping of qualified symbol tags to qualified symbols of function vars.

.. code-block:: clojure

    {my/tag my.namespace/tag-handler}

It will also search for any :external:py:class:`importlib.metadata.EntryPoint` in the group ``basilisp_data_readers`` group.
Entry points must refer to a map of data readers.
This can be disabled by setting the ``BASILISP_USE_DATA_READERS_ENTRY_POINT`` environment variable to ``false``.

.. _default_data_reader_fn:

Default Data Reader Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
By default, an exception will be raised if the reader encounters a tag that it doesn't have a data reader for.
This can be customised by binding :lpy:var:`*default-data-readers-fn*`.
It should be a function which is a function that takes two arguments, the tag symbol, and the value form.

.. _reader_special_chars:

Special Characters
------------------

Basilisp's reader has a few special characters which cause the reader to emit modified forms:

* ``'form`` is rewritten as ``(quote form)``.
* ``@form`` is rewritten to ``(basilisp.core/deref form)``.

.. _syntax_quoting:

Syntax Quoting
--------------

Syntax quoting is a facility primarily used for writing :ref:`macros` in Basilisp.
Users can syntax quote a block using the ````` character at the beginning of any valid reader form.
Within a syntax quoted form, users gain access to a few extra tools for macro writing:

* Symbols may be suffixed with a ``#`` character to have the reader generate a guaranteed-unique name to avoid name clashes during macroexpansion.
  Repeated uses of the same symbol prefix will be resolved as the same symbol name within the same syntax quoted form.
  Macros which need to generate symbols across multiple syntax quote blocks should use a :lpy:fn:`gensym` created outside both blocks and unquoted into the correct place in each.
* Forms may be injected into another form (typically a list, vector, set, or map) using the ``~`` (unquote) character.
  This is typically useful with macro parameters or other data generated external to the syntax quote.
* Sequence types may be "spliced" into the current form using the ``~@`` (unquote splice) character.
  This allows you to generate a sequence of items and have it naturally stitched into a larger syntax quoted form.

Any *unquoted* symbols not suffixed with ``#`` within a syntax quoted form will be fully resolved against the current runtime environment.
More specifically:

* Any unquoted symbol with a namespace alias will be converted into a symbol with the alias resolved to the "full" namespace name
* Any unquoted symbol with no namespace will have its full namespace added, if one exists, or otherwise the current namespace name will be added as the symbol's namespace

After all of the special processing has been applied to a syntax quoted form, the result is a standard quoted (unevaluated) form with all symbols resolved and any unquotes and splices applied.
In nearly all cases, this will be the return value from a macro function, which the compiler will compile the rest of the way into raw Python code.

.. warning::

   Using any of these special syntax quoting characters outside of a syntax quote context will result in a compiler error.

.. seealso::

   :ref:`macros`

.. _reader_conditionals:

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

.. seealso::

   :lpy:fn:`reader-conditional`, :lpy:fn:`reader-conditional?`

.. _python_version_reader_features:

Python Version Reader Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Basilisp includes a specialized set of reader features based on the major version of Python (e.g. 3.10, 3.11, etc.).
Because the API of Python's standard library changes significantly between versions, it can be challenging to support multiple versions at once.
In classical Python, users are forced to use conditional gates either at the top level of a module to define different function versions, or perhaps gate the logic within a function or class.
Both options incur some level of runtime cost.
The Python version features allow you to supply version specific overrides from the reader forward, meaning only the specific code for the version of Python you are using will be compiled and hit at runtime.

The version specific feature for Python 3.8 is ``:lpy38`` while the feature for Python 3.10 is ``:lpy310``.

In addition to the features that lock to specific versions, there are also "range" features that allow you to specify all Python versions before or after the specified version.
For example, to select all versions of Python 3.7 or greater, you would use ``:lpy37+``.
To select all versions of Python Python 3.8 or before, you would use ``:lpy38-``.

All versions of Python supported by the current version of Basilisp will be included in the default feature set.

Basilisp takes advantage of this in :lpy:ns:`basilisp.io`.

.. code-block:: clojure

   (defn delete-file
     "Delete the file named by ``f``.

     If ``silently`` is false or nil (default), attempting to delete a non-existent file
     will raise a ``FileNotFoundError``. Otherwise, return the value of ``silently``."
     ([f]
      (.unlink (as-path f))
      true)
     ([f silently]
      #?(:lpy37- (try
                   (.unlink (as-path f))
                   (catch python/FileNotFoundError e
                     (when-not silently
                       (throw e))))
         :lpy38+ (.unlink (as-path f) ** :missing-ok (if silently true false)))
      silently))

.. _python_platform_reader_features:

Platform Reader Features
^^^^^^^^^^^^^^^^^^^^^^^^

Basilisp includes a specialized reader feature based on the current platform (Linux, MacOS, Windows, etc.).
There exist cases where it may be required to use different APIs based on which platform is currently in use, so having a reader conditional to detect the current platform can simplify the development process across multiple platforms.
The reader conditional name is always a keyword containing the lowercase version of the platform name as reported by ``platform.system()``.
For example, if ``platform.system()`` returns the Python string ``"Windows"``, the platform specific reader conditional would be ``:windows``.
