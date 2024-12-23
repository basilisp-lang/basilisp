.. _testing:

Testing
=======

.. lpy:currentns:: basilisp.test

Basilisp includes a `PyTest <https://docs.pytest.org/>`_ plugin which supports running tests defined using the functions and macros in :lpy:ns:`basilisp.test`.
Tests should be located in a ``tests/`` directory off of the project root, as outlined in :ref:`project_structure`.
Basilisp test files should end with an ``.lpy`` suffix and the file basename should either be prefixed with ``test_`` or suffixed with ``_test``.
Tests can be executed using the :ref:`CLI <run_basilisp_tests>` or can be run directly using PyTest's provided CLI.

.. note::

   Basilisp supports executing both Basilisp and Python tests in the same test suite, so long as the Python tests are written using PyTest.

Tests can be written by wrapping your logic and assertions in a :lpy:fn:`deftest` form.
Basic test assertions are written using the :lpy:fn:`is` macro.
Tests within a ``deftest`` can be wrapped in an :lpy:fn:`testing` macro to both document the test function and to provide more informative testing output when tests fail.
For asserting repeatedly against different inputs, you can use the :lpy:fn:`are` templating function.

.. code-block:: clojure

   (ns my-project.test-core
    (:require [basilisp.test :refer [deftest is are testing]]))

   (deftest my-test
     (is true)

     (testing "false is really false"
       (is (not false))))

   (deftest test-adding
     (are [res x y] (= res (+ x y))
       3  1 2
       4  2 2
       0 -1 1)

.. _testing_path:

Testing and ``PYTHONPATH``
--------------------------

Typical Clojure projects will have parallel ``src/`` and ``test/`` folders in the project root.
Project management tooling typically constructs the Java classpath to include both parallel trees for development and only ``src/`` for deployed software.
Basilisp does not currently have such tooling, though it is planned.

The easiest solution to facilitate test discovery with Pytest (Basilisp's default test runner) is to create a ``tests`` directory:

.. code-block:: text

   tests
   └── myproject
       └── core_test.lpy

Test namespaces can then be created as if they are part of a giant ``tests`` package:

.. code-block:: clojure

   (ns tests.myproject.core-test)

Tests can be run with:

.. code-block:: shell

   $ basilisp test

----

Alternatively, you can follow the more traditional Clojure project structure by creating a ``test`` directory for your test namespaces:

.. code-block:: text

   test
   └── myproject
       └── core_test.lpy

In this case, the test namespace can start at ``myproject``:

.. code-block:: clojure

   (ns myproject.core-test)


However, the ``test`` directory must be explicitly added to the ``PYTHONPATH`` using the ``--include-path`` (or ``-p``) option when running the tests:

.. code-block:: shell

   $ basilisp test --include-path test

.. note::

   Test directory names can be arbitrary.
   By default, the test runner searches all subdirectories for tests.
   In the first example above (``tests``, a Python convention), the top-level directory is already in the ``PYTHONPATH``, allowing ``tests.myproject.core-test`` to be resolvable.
   In the second example (``test``, a Clojure convention), the test directory is explicitly added to the ``PYTHONPATH``, enabling ``myproject.core-test`` to be resolvable.

.. _test_fixtures:

Fixtures
--------

Basilisp supports test fixtures which can serve as setup and teardown functions for either individual tests or for whole test modules.
Fixtures can be applied using the :lpy:fn:`use-fixtures` function.

Basilisp comes with one builtin fixture, which can generate a temporary directory for the duration of the test.

.. code-block:: clojure

   (ns my-project.test-core
     (:require
      [basilisp.test :as test :refer [deftest is are testing]]
      [basilisp.test.fixtures :as fixtures :refer [*tempdir*]))

   (test/use-fixtures :each fixtures/tempdir)

   (deftest some-test
     ;; accessing ``*tempdir*`` here will give a directory that will be
     ;; cleaned up after this test is run
     )

Fixtures can trivially be written by writing a basic function and passing it to ``use-fixtures``.
For fixtures which only need to perform setup, a fixture of no arguments will suffice.
For fixtures which must perform setup and teardown or just teardown, a function of no arguments should be written and it should :lpy:form:`yield` after the setup step and before the teardown.
The test framework will yield control back to the fixture function when it is time to teardown.

You can see below that the fixture uses a :ref:`dynamic Var <dynamic_vars>` to communicate what it has done back to any tests that use this fixture.

.. code-block::

   (def ^:dynamic *tempdir* nil)

   (defn tempdir
     []
     (with-open [d (tempfile/TemporaryDirectory)]
       (binding [*tempdir* d]
         (yield))))

.. warning::

   Basilisp test fixtures are not related to PyTest fixtures and they cannot be used interchangeably.
