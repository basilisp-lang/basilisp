.. _contributing:

Contributing
============

Thank you for your interest in contributing!
Contributions to the project are gladly accepted.
We accept the following types of contributions:

* **Bug Reports:** Submit a new issue on GitHub reporting a bug with Basilisp.
  Contributors are encouraged to first search the existing issues to see if the bug has previously been reported or is currently being addressed.
  Bugs may sometimes be fixed in unreleased code, so be sure to look at the `CHANGELOG <https://github.com/basilisp-lang/basilisp/blob/main/CHANGELOG.md>`_ in the ``main`` branch to see if the fix has simply not yet been released.

* **Feature Requests:** Submit a new issue on GitHub reporting a feature request.
  Because Basilisp targets a high degree of compatibility with Clojure, feature requests should generally be for features from Clojure which have not yet been implemented in Basilisp.
  Occasionally, new features outside of the scope of Clojure compatibility may be covered, in particular if the feature is related to a Python ecosystem affordance.

* **Documentation:** Documentation is hard.
  Basilisp strives to include correct and complete documentation, but disconnects between the code and text do happen.
  File an issue on GitHub if you find incorrect or incomplete documentation.
  If you're feeling brave, consider opening a PR to address this issue as well.

* **PRs:** Submit a new pull request fixing an issue, implementing a new feature, or updating some project documentation.
  Potential contributors are encouraged to reach out to project maintainers first before addressing an issue to ensure that their PR can be reviewed and accepted for an issue.
  Before opening a PR, please review the :ref:`documentation <developing_on_basilisp>` on how to develop on Basilisp.

.. note::

   Please note that Basilisp is still a personal project for its maintainers and we make no guarantees of PR review turnaround time or acceptance.

.. _developing_on_basilisp:

Developing on Basilisp
----------------------

.. _development_requirements:

Requirements
^^^^^^^^^^^^

This project uses `Poetry <https://github.com/python-poetry/poetry>`_ to manage the Python virtual environment, project dependencies, and package publication.
See the instructions on that repository to install in your local environment.
Because Basilisp is intended to be used as a library, no ``poetry.lock`` file is committed to the repository.
Developers should generate their own lock file and update it regularly during development instead.

Additionally, `pyenv <https://github.com/pyenv/pyenv>`_ is recommended to manage versions of Python readily on your local development environment.
Setup of ``pyenv`` is somewhat more specific to your environment, so see the documentation in the repository for more information.

.. _getting_started_development:

Getting Started
^^^^^^^^^^^^^^^

To prepare your `poetry` environment, you need to install dependencies:

.. code-block:: bash

   poetry install

Afterwards, you can start up the REPL for development with a simple:

.. code-block:: bash

   make repl

.. _linting_testing_and_type_checking:

Linting, Running Tests, and Type Checking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Basilisp automates linting, running tests, and type checking using `Tox <https://github.com/tox-dev/tox>`_.
All three steps can be performed using a provided ``make`` target:

.. code-block:: bash

   make test

Testing is performed using `PyTest <https://github.com/pytest-dev/pytest/>`_.
Type checking is performed by `MyPy <http://mypy-lang.org/>`_.
Linting is performed using `Prospector <https://prospector.landscape.io/en/master/>`_.
Formatting is performed using `Black <https://github.com/psf/black>`_.

New *code* contributions should include test coverage covering all new branches unless otherwise directed by the project maintainers.