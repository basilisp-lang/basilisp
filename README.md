# 🐍 basilisp 🐍

A Lisp dialect inspired by Clojure targeting Python 3.

**Disclaimer:** _Basilisp is a project I created to learn about Python, Clojure,
and hosted languages generally. It should not be used in a production setting._

![PyPI](https://img.shields.io/pypi/v/basilisp.svg?style=flat-square) ![python](https://img.shields.io/pypi/pyversions/basilisp.svg?style=flat-square) ![CircleCI](https://img.shields.io/circleci/project/github/chrisrink10/basilisp.svg?style=flat-square) ![Coveralls github](https://img.shields.io/coveralls/github/chrisrink10/basilisp.svg?style=flat-square) ![license](https://img.shields.io/github/license/chrisrink10/basilisp.svg?style=flat-square)

## Getting Started

Basilisp is developed on [GitHub](https://github.com/chrisrink10/basilisp)
and hosted on [PyPI](https://pypi.python.org/pypi/basilisp). You can
fetch Basilisp using a simple:

```bash
pip install basilisp
```

Once Basilisp is installed, you can enter into the REPL using:

```bash
python -m basilisp.main
```

Basilisp features many of the same functions and idioms as [Clojure](https://clojure.org/)
so you may find guides and documentation there helpful for getting
started.

## Developing on Basilisp

### Requirements

This project uses [`pipenv`](https://github.com/kennethreitz/pipenv) to
manage the Python virtual environment and project dependencies. `pipenv`
can be installed using Homebrew (on OS X) or `pip` otherwise:

```bash
brew install pipenv
```

```bash
pip install --user pipenv
```

Additionally, [`pyenv`](https://github.com/pyenv/pyenv) is recommended to 
manage versions of Python readily on your local development environment.
Setup of `pyenv` is somewhat more specific to your environment, so see
the documentation in the repository for more information.

### Performing Type Checking and Linting

Perform type checking with [MyPy](http://mypy-lang.org/) using:

```bash
make typecheck
```

Perform linting with [Prospector](https://prospector.landscape.io/en/master/):

```bash
make lint
```

The linting artifact will be emitted at the project root as `lintout.txt`.

### Running Tests

Tests can be run using the following command:

```bash
make test
```

## License

MIT License
