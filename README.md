# 🐍 basilisp 🐍

A Lisp dialect inspired by Clojure targeting Python 3.

**Disclaimer:** _Basilisp is a project I created to learn about Python, Clojure,
and hosted languages generally. It should not be used in a production setting._

[![PyPI](https://img.shields.io/pypi/v/basilisp.svg?style=flat-square)](https://pypi.org/project/basilisp/) [![python](https://img.shields.io/pypi/pyversions/basilisp.svg?style=flat-square)](https://pypi.org/project/basilisp/) [![pyimpl](https://img.shields.io/pypi/implementation/basilisp.svg?style=flat-square)](https://pypi.org/project/basilisp/) [![readthedocs](https://img.shields.io/readthedocs/basilisp.svg?style=flat-square)](https://basilisp.readthedocs.io/) [![CircleCI](	https://img.shields.io/circleci/project/github/basilisp-lang/basilisp/master.svg?style=flat-square)](https://circleci.com/gh/basilisp-lang/basilisp) [![Coveralls github](https://img.shields.io/coveralls/github/basilisp-lang/basilisp.svg?style=flat-square)](https://coveralls.io/github/basilisp-lang/basilisp) [![license](https://img.shields.io/github/license/basilisp-lang/basilisp.svg?style=flat-square)](https://github.com/basilisp-lang/basilisp/blob/master/LICENSE)

## Getting Started

Basilisp is developed on [GitHub](https://github.com/chrisrink10/basilisp)
and hosted on [PyPI](https://pypi.python.org/pypi/basilisp). You can
fetch Basilisp using a simple:

```bash
pip install basilisp
```

Once Basilisp is installed, you can enter into the REPL using:

```bash
basilisp repl
```

Basilisp [documentation](https://basilisp.readthedocs.io) can help guide your 
exploration at the REPL. Additionally, Basilisp features many of the same functions 
and idioms as [Clojure](https://clojure.org/) so you may find guides and 
documentation there helpful for getting started.

## Developing on Basilisp

### Requirements

This project uses [`poetry`](https://github.com/python-poetry/poetry) to manage
the Python virtual environment, project dependencies, and package publication.
See the instructions on that repository to install in your local environment.
Because `basilisp` is intended to be used as a library, no `poetry.lock` file
is committed to the repository. Developers should generate their own lock file
and update it regularly during development instead.

Additionally, [`pyenv`](https://github.com/pyenv/pyenv) is recommended to 
manage versions of Python readily on your local development environment.
Setup of `pyenv` is somewhat more specific to your environment, so see
the documentation in the repository for more information.

### Getting Started

To prepare your `poetry` environment, you need to install dependencies:

```bash
poetry install
```

Afterwards, you can start up the REPL for development with a simple:

```bash
make repl
```

### Linting, Running Tests, and Type Checking

Basilisp automates linting, running tests, and type checking using 
[Tox](https://github.com/tox-dev/tox). All three steps can be performed
using a simple `make` target:

```bash
make test
```

Testing is performed using [PyTest](https://github.com/pytest-dev/pytest/). 
Type checking is performed by [MyPy](http://mypy-lang.org/). Linting is 
performed using [Prospector](https://prospector.landscape.io/en/master/).

## License

Eclipse Public License 1.0
