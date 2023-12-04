# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
 * Fix issue with `case` evaluating all of its clauses expressions (#699).
 * Fix issue with relative paths dropping their first character on MS-Windows (#703).
 * Fix incompatibility with `(str nil)` returning "nil" (#706).
 * Fix `sort-by` support for maps and boolean comparator fns (#709).
 * Fix `sort` support for maps and boolean comparator fns (#711).
 * Fix `(is (= exp act))` should only evaluate its args once on failure (#712).
 * Fix issue with `with` failing with a traceback error when an exception is thrown (#714).
 * Fix issue with `sort-*` family of funtions returning an error on an empty seq (#716).

## [v0.1.0a2]
### Added
 * Added support for fixtures in `basilisp.test` (#654)
 * Added support for Python 3.10 and 3.11 (#659, #693)
 * Added a Sphinx autodoc plugin for generating API documentation for Basilisp namespaces (#658)
 * Added support for rewriting required namespaces starting with `clojure.` as `basilisp.` if the original name isn't found on the import path (#670, #676)
 * Added support for inlining simple functions (#673)
 * Added the `clojure.core` functions from v1.11 (#672)

### Changed
 * Set tighter bounds on dependency version ranges (#657)
 * Improved on and completed several different sections of the documentation (#661, #669)
 * Delete unused utility functions after they are generated and executed by the REPL to save memory (#674)

### Fixed
 * Fixed the `with` macro definition to match the Python language spec (#656)
 * Fixed a bug where `py->lisp` did not convert map keys or values into Basilisp objects (#679)

### Other
 * Run CPython CI checks on Github Actions rather than CircleCI (#683)
 * Remove support for Python 3.6 and 3.7, which are both EOL (#691)
 * Fix test suite failures on Windows and add Github Actions runners for testing on Windows (#688)
 * Update Prospector version for linting (#694)

## [v0.1.0a1]
### Added
 * Added a bootstrapping function for easily bootstrapping Basilisp projects from Python (#620)
 * Added support for watchers and validator functions on Atoms and Vars (#627)
 * Added support for Taps (#631)
 * Added support for hierarchies (#633)
 * Added support for several more utility Namespace and Var utility functions (#636)
 * Added `basilisp.io` namespace with polymorphic reader and writer functions (#645)
 * Added support for coroutines and generators using `yield` syntax (#652)

### Changed
 * PyTest is now an optional extra dependency, rather than a required dependency (#622)
 * Generated Python functions corresponding to nested functions are now prefixed with the containing function name, if one exists (#632)
 * `basilisp.test/are` docstring now indicates that line numbers may be suppressed on assertion failures created using `are` (#643)
 * Multimethods now support providing a custom hierarchy and dispatch to registered values using `isa?` (#644)

### Fixed
 * Fixed a bug where `seq`ing co-recursive lazy sequences would cause a stack overflow (#632)
 * Fixed a spurious failure in the test runner and switched to using macro forms for test line numbers (#631)
 * Fixed a bug that allowed dynamic Vars to be `set!` even if they weren't thread-bound (#638)
 * Fixed a bug where it was impossible to specify negative CLI options for the compiler flags (#638)
 * Fixed a bug where it was impossible to use more than a single body expression in a `try` special form (#640)
 * Fixed a bug where re-`def`ing a Var (regardless of `^:redef` metadata) would not update metadata or dynamic flag (#642)
 * Fixed a bug where private Vars could be resolved from the source namespace of a public macro during macroexpansion (#648)
 * Fixed a bug where trailing quotes were not allowed in Symbols and Keywords (#650)

### Removed
 * Removed Click as a dependency in favor of builtin `argparse` (#622, #624, #636)
 * Removed Atomos as a dependency in favor of `readerwriterlock` (#624)

## [v0.1.dev15]
### Added
 * Added support for auto-resolving namespaces for keyword from the current namespace using the `::kw` syntax (#576)
 * Added support for namespaced map syntax (#577)
 * Added support for numeric constant literals for NaN, positive infinity, and negative infinity (#582)
 * Added `*basilisp-version*` and `*python-version*` Vars to `basilisp.core` (#584)
 * Added support for function decorators to `defn` (#585)
 * Added the current Python version (`:lpy36`, `:lpy37`, etc.) as a default reader feature for reader conditionals (#585)
 * Added default reader features for matching Python version ranges (`:lpy36+`, `:lpy38-`, etc.) (#593)
 * Added `lazy-cat` function for lazily concatenating sequences (#588)
 * Added support for writing EDN strings from `basilisp.edn` (#600)
 * Added a persistent queue data type (#606)
 * Added support for transducers (#601)
 * Added support for Python 3.9 (#608)
 * Added support for 3-way comparators (#609)

### Changed
 * Moved `basilisp.lang.runtime.to_seq` to `basilisp.lang.seq` so it can be used within that module and by `basilisp.lang.runtime` without circular import (#588)
 * Keyword hashes are now pre-computed when they are created, so they do not need to be recomputed again to be fetched from the intern cache (#592)
 * The compiler now uses the pre-computed hash to lookup keywords directly, which should improve lookup time for repeated invocations (#592)
 * Symbol hashes are now pre-computed when they are created (#592)
 * Moved `basilisp.core.template` to `basilisp.template` to match Clojure (#599)
 * Refactor compiler to use `functools.singledispatch` for type based dispatch (#605)
 * Rename `List`, `Map`, `Set`, and `Vector` to `PersistentList`, `PersistentMap`, `PersistentSet`, and `PersistentVector` respectively (#605)

### Fixed
 * Fixed a bug where `def` forms did not permit recursive references to the `def`'ed Vars (#578)
 * Fixed a bug where `concat` could cause a `RecursionEror` if used on a `LazySeq` instance which itself calls `concat` (#588)
 * Fixed a bug where map literals in function reader macro forms caused errors during reading (#599)
 * Fixed a bug where `some->` and `some->>` threading macros would thread `nil` first arguments (#599)

### Removed
 * Removed `pyfunctional` dependency in favor of Python standard library functions (#589)

### Other
 * Basilisp uses `poetry` for dependency and virtual environment management, as well as for publishing to PyPI (#616)

## [v0.1.dev14] - 2020-06-18
### Added
 * Added support for `future`s (#441)
 * Added support for calling Python functions and methods with keyword arguments (#531)
 * Added support for Lisp functions being called with keyword arguments (#528)
 * Added support for multi-arity methods on `deftype`s (#534)
 * Added metadata about the function or method context of a Lisp AST node in the `NodeEnv` (#548)
 * Added `reify*` special form (#425)
 * Added support for multi-arity methods on `definterface` (#538)
 * Added support for Protocols (#460)
 * Added support for Volatiles (#460)
 * Added JSON encoder and decoder in `basilisp.json` namespace (#484)
 * Added support for generically diffing Basilisp data structures in `basilisp.data` namespace (#555)
 * Added support for artificially abstract bases classes in `deftype`, `defrecord`, and `reify` types (#565)
 * Added support for transient maps, sets, and vectors (#568)

### Changed
 * Basilisp set and map types are now backed by the HAMT provided by `immutables` (#557)
 * `get` now responds `nil` (or its default) for any unsupported types (#570)
 * `nth` now supports only sequential collections (or `nil`) and will throw an exception for any invalid types (#570)
 * Use `functools.singledispatch` for to achieve higher performance polymorphism on most runtime functions (#552, #559)
 * Update the keyword cache to use a Python `threading.Lock` rather than an Atom (#552)
 * `rest` no longer returns `nil`, it always returns an empty sequence (#558)

### Fixed
 * Fixed a bug where the Basilisp AST nodes for return values of `deftype` members could be marked as _statements_ rather than _expressions_, resulting in an incorrect `nil` return (#523)
 * Fixed a bug where `defonce` would throw a Python SyntaxError due to a superfluous `global` statement in the generated Python (#525)
 * Fixed a bug where Basilisp would throw an exception when comparing seqs by `=` to non-seqable values (#530)
 * Fixed a bug where aliased Python submodule imports referred to the top-level module rather than the submodule (#533)
 * Fixed a bug where static methods and class methods on types created by `deftype` could not be referred to directly (defeating the purpose of the static or class method) (#537)
 * Fixed a bug where `deftype` forms could not be declared without at least one field (#540)
 * Fixed a bug where not all builtin Basilisp types could be pickled (#518)
 * Fixed a bug where `deftype` forms could not be created interfaces declared not at the top-level of a code block in a namespace (#376)
 * Fixed multiple bugs relating to symbol resolution of `import`ed symbols in various contexts (#544)
 * Fixed a bug where the `=` function did not respect the equality partition for various builtin collection types (#556)
 * Fixed a bug where collection types could evaluate as boolean `false` (#566)
 * Fixed a bug where `reduce` required a 1-arity function for the variant with an initial value, rather than returning that initial value (#567)

## [v0.1.dev13] - 2020-03-16
### Added
 * Added support for Shebang-style line comments (#469)
 * Added multiline REPL support using `prompt-toolkit` (#467)
 * Added node syntactic location (statement or expression) to Basilisp AST nodes emitted by the analyzer (#463)
 * Added `letfn` special form (#473)
 * Added `defn-`, `declare`, and `defonce` macros (#480)
 * Added EDN reader in the `basilisp.edn` namespace (#477)
 * Added line, column, and file information to reader `SyntaxError`s (#488)
 * Added context information to the `CompilerException` string output (#493)
 * Added Array (Python list) functions (#504, #509)
 * Added shell function in `basilisp.shell` namespace (#515)
 * Added `apply-template` function to `basilisp.core.template` namespace (#516)

### Changed
 * Change the default user namespace to `basilisp.user` (#466)
 * Changed multi-methods to use a `threading.Lock` internally rather than an Atom (#478)
 * Changed the Basilisp module type from `types.ModuleType` to a custom subtype with support for custom attributes (#482)
 * Basilisp's runtime function `Namespace.get_or_create` no longer refers `basilisp.core` by default, which allows callers to exclude `basilisp.core` names in the `ns` macro (#481)
 * Namespaces now use a single internal lock rather than putting each property inside of an Atom (#494)
 * Refactor the testrunner to use fewer `atom`s in `basilisp.test` (#495)

### Fixed
 * Fixed a reader bug where no exception was being thrown splicing reader conditional forms appeared outside of valid splicing contexts (#470)
 * Fixed a bug where fully Namespace-qualified symbols would not resolve if the current Namespace did not alias the referenced Namespace (#479)
 * Fixed a bug where the `quote` special form allowed more than one argument and raised an unintended exception when no argument was provided (#497)
 * Fixed a bug where compiler options specified via command-line argument or environment variable were not honored by the importer (#507)
 * Fixed a bug where private Vars from other Namespaces could be referenced if the Namespace was aliased when it was required (#514)
 * Fixed a bug where collections with trailing end tokens separated from the collection only by a comment (#520)

## [v0.1.dev12] - 2020-01-26
### Added
 * Added new control structures: `dotimes`, `while`, `dorun`, `doall`, `case`, `for`, `doseq`, `..`, `with`, `doto` (#431)
 * Added `basilisp.walk` namespace with generic tree-walker functions (#434)
 * Added several new higher-order functions (#433)
 * Added `basilisp.template` namespace with templating utility functions (#433)
 * Added `basilisp.test/are` for writing multiple similar assertions (#433)
 * Added many new collection and sequence functions (#439)
 * Added support for Promises (#440)
 * Added support for Python 3.8 (#447)
 * Added `vary-meta`, `alter-meta!`, and `reset-meta!` utility functions (#449)
 * Added support for the primitive type coercion API (#451)
 * Added support for the unchecked arithmetic API (#452)
 * Added a `Makefile` utility for generating the Python code for `basilisp.core` (#456)

### Changed
 * Compile `attrs` instances internally without `cmp` keyword argument for `attrs` >= 19.2.0 (#448)
 * Small Python changes in the compiler to remove redundant operations (#450)

### Fixed
 * Fixed an issue where `macroexpand` and `macroexpand-1` attempted to resolve symbols rather than leaving them unresolved (#433)
 * Fixed an issue with transient macro namespace symbol resolution in the analyzer (#438)
 * Fixed a bug where `ISeq` iterators could stack overflow for long sequences (#445)
 * Fixed the `importer` test suite's use of `multiprocessing` for Python 3.8 (#446)
 * Correct the `with-meta` interface and replace incorrect usages with `vary-meta` (#449)
 * Fixed an issue where line/column metadata was not properly being fetched by the analyzer (#454)
 * Warnings for Basilisp code now indicate namespace and line number of that code (#457)
 * Errors resolving nested symbols are now much more helpful (#459)

## [v0.1.dev11] - 2019-07-27
### Added
 * `macroexpand` and `macroexpand-1` functions (#394)
 * `reset-vals!` and `swap-vals!` Atom functions (#399)
 * `format`, `printf`, and a few other formatting functions (#401)
 * `rand`, `rand-int`, and a few other basic random functions (#402)
 * `assoc-in`, `update-in`, and a few other associative utility functions (#404)
 * Several namespace utility functions (#405)
 * Preliminary support for Python 3.8 (#406 and #407)
 * Added the `ILookup` interface to `IAssociative` (#410)
 * Support for Reader Conditional syntax (#409)

### Changed
 * Python builtins may now be accessed using the `python` namespace from within Basilisp code, rather than `builtins` (#400)
 * Basilisp code files can now be named and organized more like Clojure projects. `__init__` files are not necessary, though they may still be used. Folders can bear the same name as a namespace file, which will allow nesting. (#393)
 * Renamed Basilisp's parser module to analyzer, to more accurately reflect it's purpose (#390)
 * Changed `binding` behavior to use `push-thread-bindings` and `pop-thread-bindings`, which use a thread-local for bindings, to more closely emulate Clojure (#405)
 * Internal `IAsssociative.entry` usages have been changed to `ILookup.val_at` (#410)

### Fixed
 * Allow direct code references to static methods and fields (#392)

## [v0.1.dev10] - 2019-05-06
### Added
 * Added support for Record data types (#374, #378, and #380)
 * Lots more useful core library functions (#373)

### Changed
 * Refactor core interfaces under `basilisp.lang.interfaces` module (#370)
 * Organize interface hierarchy to match Clojure's (#372)
 * Compile qualified `basilisp.lang.*` module references down to aliased references (#366)
 * `let` and `loop` forms may now have empty binding vectors and empty bodies (#382)

## [v0.1.dev9] - 2019-03-29
### Added
 * Add support for custom data types (#352)
 * Add an environment variable which allows users to disable emitting Python AST strings (#356)
 * Functions now support metadata (#347)
 * `defn` forms can attach metadata to functions via an attribute map (#350)
 * Basilisp code can be executed as a script via the CLI from standard in (#349)
 * Support for `async` functions (via `:async` metadata or `defasync` form) and new `await` special form (#342)
 * Symbol and keyword completion at the REPL if `readline` is available (#340)
 * Macro environment is now being passed as the first argument of macros (#339)
 * Create Python literals using `#py` reader tag and corresponding Basilisp data structure (#337)

### Fixed
 * Nested Python imports can no longer be obscured by their parent module name (#360)

## [v0.1.dev8] - 2019-03-10
### Added
 * Basilisp compiler parses Basilisp code into an intermediate AST prior to Python code generation (#325)
 * Add meta to `def` forms in the compiler (#324)
 * Switch builtin exceptions to use `attrs` (#334)

### Fixed
 * Quoted interop forms and properties no longer emit errors (#326)
 * Improve test suite performance by using a mutable `dict` for `lrepr` kwargs generation (#331)
 * Attach line and column metadata to functions, quoted, and deref forms (#333)
 * Log internal `nth` messages as `TRACE` rather than `DEBUG` (#335)

## [v0.1.dev7] - 2018-12-18
### Added
 * Add `loop` special form (#317)
 * Add Lisp `repr` support using `LispObject` abstract base class (#316)
 * Python imports may be aliased like `require` using a vector (#319)
 * The compiler emits a warning when a local symbol is unused (#314)
 * The compiler can optionally emit a warning when a local name or Var name is shadowed (#303, #312)
 * The compiler can optionally emit a warning when a Var reference requires indirection (#306)

### Fixed
 * `basilisp.core/ns-resolve` can now resolve aliased symbols (#313)
 * Python interop calls can now be made within threading macros (#308)

### Other
 * Basilisp source code is now released under the [EPL 1.0](https://www.eclipse.org/legal/epl-v10.html)

## [v0.1.dev6] - 2018-11-02
### Added
 * Print functions respect output stream bound to `*out*` (#257)
 * Capture line numbers for test failures and errors (#270)
 * Support threading macros (#284)
 * Support redefinable Vars and suppressing warnings when Vars are redefined (#292)
 * Add more core functions (#293, #296, #298)
 * Support destructuring of arguments in `fn` and bindings in `let` (#289)
 * Add set library functions (#261)

### Fixed
 * Keyword string and repr are now identical (#259)
 * Don't attempt to resolve names inside syntax-quote (#272)
 * Properly resolve names inside `catch` blocks (#273)
 * Loosen syntax requirements for interop properties (#276)
 * Fix `basilisp.core/map` lazy sequence behavior (#278)
 * Allow empty function definitions (#280)
 * Allow the dollar sign `$` in symbols (#286)
 * Report exceptions caught in the test runner as errors (#300)

## [v0.1.dev5] - 2018-10-14
### Added
 * Add a logger to Basilisp library code (with a default NullHandler) (#243)
 * Implement Bytecode caching for compiled Basilisp modules (#244)

## [v0.1.dev4] - 2018-09-29
### Changed
 * Bump version to support Python 3.7 on PyPI

## [v0.1.dev3] - 2018-09-29
### Added
 * Add multi-methods to Basilisp (#222)
 * The Basilisp PyTest runner now prints exception messages for non-failure errors (#235)
 * Add a few new namespace functions to `basilisp.core` (#224)
 * Create namespace setting context managers for use in tests (#228)

### Fixed
 * Removed usages of `seq.grouped` from PyFunctional which throw `StopIteration` exceptions and cause Basilisp to be unusable with Python 3.7 (#221)
 * Fix a bug where the wrong namespace may be set in `deftest` defined tests (#232)
 * Fix a bug where non-symbols could not appear in the member position of Python interop special forms (#230)
 * Fix several instances of bugs where Basilisp tests were changing `basilisp.core/*ns*` without resetting it, which would cause cascading test failures (#228)

## [v0.1.dev2] - 2018-09-27
### Added
 * Add PyTest runner tests written using the `basilisp.test/deftest` macro (#195)
 * Throw a useful error when no Var is bound to a symbol (#197)
 * Add a string library as `basilisp.string` (#187)
 * Add regex functions to `basilisp.core` (#193)
 * Add namespace functions to `basilisp.core` (#176)
 * The reader can return a custom EOF indicator (#218)

### Fixed
 * Fixed a bug where comment literals were not be fully removed from reader outputs (#196)
 * Fixed a bug where comment literals caused syntax errors inside collection literals (#196)
 * Imported namespaces no longer create extra namespaces bound to munged Python module names (#216)
 * Fixed a bug where `import*`s would not work within other forms 
 * Fixed a bug where the Basilisp import hook could be added to `sys.meta_path` multiple times (#213)
 * Fixed a bug where keywords could not be used in function position (#174)
 * Fixed a bug where macro symbols were not resolved using the same heuristics as other symbols (#183)
 * Fixed a bug where lazy sequences were not resolved even if they were empty-checked (#182)
 * Fixed a bug where new namespaces were created whenever the compiler checked if a namespace existed to resolve a symbol (#211)
 * Fixed a bug where Python string escape sequences were not handled correctly by the reader (#192)
 * Fixed a bug where character literals caused a syntax error inside collections (#192)

## [v0.1.dev1] - 2018-09-13
### Added
 * Basic CLI for REPL and running scripts
 * REPL convenience functions `doc` and `pydoc` will return Basilisp and Python documentation respectively
 * Vars containing the last 3 expression results (`*1`, `*2`, `*3`) and last exception (`*e`) are now included on the REPL
 * Added support for [Decimal](https://docs.python.org/3/library/decimal.html), [Fraction](https://docs.python.org/3/library/fractions.html), and [complex](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex) types
 * Support for Clojure style object instantiation syntax: `(new com.lisp.Object)` or `(com.list.Object.)`
 * Read/eval functions in `basilisp.core`
 * Support for customizing data readers either by binding `*data-readers*` or supplying a keyword argument from Python
 * Support for character literals using `\a` syntax
 * Support for deref-literals using `@`

### Fixed
 * Dynamic vars are now properly compiled as dynamic, allowing thread-local bindings
 * `let*` bindings can no longer eagerly evaluate binding expressions in conditional branches which will not be taken
 * `catch` expressions no longer throw an error when they appear in a syntax-quote
 * Basilisp files imported using the Basilisp import hook now properly resolve symbols in syntax quotes
 * Basilisp's `Map` type now supports `cons`ing other maps.
 * Syntax quoted special forms are no longer resolved into namespaced symbols

## [v0.1.dev0] - 2018-08-31
### Added
- Basilisp language and compiler base.

[v0.1.0a2]: https://github.com/chrisrink10/basilisp/compare/v0.1.0a1..v0.1.0a2
[v0.1.0a1]: https://github.com/chrisrink10/basilisp/compare/v0.1.dev15..v0.1.0a1
[v0.1.dev15]: https://github.com/chrisrink10/basilisp/compare/v0.1.dev14..v0.1.dev15
[v0.1.dev14]: https://github.com/chrisrink10/basilisp/compare/v0.1.dev13..v0.1.dev14
[v0.1.dev13]: https://github.com/chrisrink10/basilisp/compare/v0.1.dev12..v0.1.dev13
[v0.1.dev12]: https://github.com/chrisrink10/basilisp/compare/v0.1.dev11..v0.1.dev12
[v0.1.dev11]: https://github.com/chrisrink10/basilisp/compare/v0.1.dev10..v0.1.dev11
[v0.1.dev10]: https://github.com/chrisrink10/basilisp/compare/v0.1.dev9..v0.1.dev10
[v0.1.dev9]: https://github.com/chrisrink10/basilisp/compare/v0.1.dev8..v0.1.dev9
[v0.1.dev8]: https://github.com/chrisrink10/basilisp/compare/v0.1.dev7..v0.1.dev8
[v0.1.dev7]: https://github.com/chrisrink10/basilisp/compare/v0.1.dev6..v0.1.dev7
[v0.1.dev6]: https://github.com/chrisrink10/basilisp/compare/v0.1.dev5..v0.1.dev6
[v0.1.dev5]: https://github.com/chrisrink10/basilisp/compare/v0.1.dev4..v0.1.dev5
[v0.1.dev4]: https://github.com/chrisrink10/basilisp/compare/v0.1.dev3..v0.1.dev4
[v0.1.dev3]: https://github.com/chrisrink10/basilisp/compare/v0.1.dev2..v0.1.dev3
[v0.1.dev2]: https://github.com/chrisrink10/basilisp/compare/v0.1.dev1..v0.1.dev2
[v0.1.dev1]: https://github.com/chrisrink10/basilisp/compare/v0.1.dev0..v0.1.dev1
[v0.1.dev0]: https://github.com/chrisrink10/basilisp/releases/tag/v0.1.dev0
