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