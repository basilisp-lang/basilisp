# 🐍 basilisp 🐍

A Lisp dialect inspired by Clojure targeting Python.

![CircleCI](https://img.shields.io/circleci/project/github/chrisrink10/basilisp.svg?style=flat-square) ![Coveralls github](https://img.shields.io/coveralls/github/chrisrink10/basilisp.svg?style=flat-square)

## Requirements

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

## Testing

Tests can be run using the following command:

```bash
make test
```

## License

MIT License
