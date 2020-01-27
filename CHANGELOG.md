# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
 * Added support for Shebang-style line comments (#469)
 * Added node syntactic location (statement or expression) to Basilisp AST nodes emitted by the analyzer (#463)

### Changed
 * Change the default user namespace to `basilisp.user` (#466)

### Fixed
 * Fixed a reader bug where no exception was being thrown splicing reader conditional forms appeared outside of valid splicing contexts (#470)

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
