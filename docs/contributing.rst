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
Developers should install all supported versions of Python and configure them for the project using ``pyenv local`` in the Basilisp source directory.
The ``.python-version`` file is included in the project ``.gitignore``.

.. _getting_started_development:

Getting Started
^^^^^^^^^^^^^^^

To prepare your `poetry` environment, you need to install dependencies:

.. code-block:: bash

   poetry install

Afterwards, you can open a new Poetry shell to start up the REPL for development.
The ``make repl`` target _may_ be sufficient for local development, though developers working on the Basilisp compiler or standard library are encouraged to enable a more verbose set of configurations for detecting issues during development.
The command below enables the highest level of logging and disables namespace caching, both of which can help reveal otherwise hidden issues during development.

.. code-block:: bash

   poetry shell
   BASILISP_USE_DEV_LOGGER=true BASILISP_LOGGING_LEVEL=TRACE BASILISP_DO_NOT_CACHE_NAMESPACES=true basilisp repl

Developers working on the Basilisp compiler should periodically update their dependencies to ensure no incompatibilities have been introduced into the codebase with new dependency versions.

.. code-block:: bash

   poetry update

.. _linting_testing_and_type_checking:

Linting, Running Tests, and Type Checking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Basilisp automates linting, running tests, and type checking using `Tox <https://github.com/tox-dev/tox>`_.
All three steps can be performed across all supported versions of CPython using a provided ``make`` target:

.. code-block:: bash

   make test

To run a more targeted CI check directly from within the Poetry shell, developers can use ``tox`` commands directly.
For instance, to run only the tests for ``basilisp.io`` on Python 3.12, you could use the following command:

.. code-block:: bash

   tox run -e py312 -- tests/basilisp/test_io.lpy

Developers are encouraged to investigate the available configurations in ``tox.ini`` to determine which CI targets they will have at their disposal.

Testing is performed using `PyTest <https://github.com/pytest-dev/pytest/>`_.
Type checking is performed by `MyPy <http://mypy-lang.org/>`_.
Linting is performed using `Prospector <https://prospector.landscape.io/en/master/>`_.
Formatting is performed using `Black <https://github.com/psf/black>`_.

New *code* contributions should include test coverage covering all new branches unless otherwise directed by the project maintainers.