.. _cli:

CLI
===

Basilisp includes a basic CLI tool which can be used to start a local :ref:`repl` session, run some code as a string or from a file, or execute the test suite using the builtin PyTest integration.

.. _configuration:

Configuration
-------------

Basilisp exposes all of it's available configuration options as CLI flags and environment variables, with the CLI flags taking precedence.
All Basilisp CLI subcommands which include configuration note the available configuration options when the ``-h`` and ``--help`` flags are given.
Generally the Basilisp CLI configuration options are simple passthroughs that correspond to :ref:`configuration options for the compiler <compiler_configuration>`.

.. _start_a_repl_session:

Start a REPL Session
--------------------

Basilisp's CLI includes a basic REPL client powered using `Prompt Toolkit <https://github.com/prompt-toolkit/python-prompt-toolkit>`_ (and optionally colored using `Pygments <https://pygments.org/>`_ if you installed the ``pygments`` extra).
You can start the local REPL client with the following command.

.. code-block:: bash

   basilisp repl

The builtin REPL supports basic code completion suggestions, syntax highlighting (if Pygments is installed), multi-line editing, and cross-session history.

.. note::

   You can exit the REPL by entering an end-of-file ("EOF") character by pressing Ctrl+D at your keyboard.

.. _start_an_nREPL_session:

Start an nREPL Session
----------------------

Basilisp's CLI incorporates an nREPL server adapted from `nbb <https://github.com/babashka/nbb>`_.

To start the server from the command line use the following command

.. code-block:: bash

   basilisp nrepl-server
   # => nREPL server started on port 50407 on host 127.0.0.1 - nrepl://127.0.0.1:50407

You can then establish a connection from your IDE to the server address.

- from `Emacs`, using `CIDER <https://github.com/clojure-emacs/cider>`_

  .. code-block::

     M-x cider-connect-clj

- from `Visual Studio Code`, using `Calva <https://calva.io/>`_

  .. code-block::

     REPL -> Connect to a running REPL in your project -> Generic

.. _run_basilisp_code:

Run Basilisp Code
-----------------

You can run Basilisp code from a string or by directly naming a file with the CLI as well.

.. code-block:: bash

   basilisp run -c '(+ 1 2 3)'

.. code-block:: bash

   basilisp run path/to/some/file.lpy

Any arguments passed to ``basilisp run`` beyond the name of the file or the code string will be bound to the var :lpy:var:`*command-line-args*` as a vector of strings.
If no arguments are provided, ``*command-line-args*`` will be ``nil``.

.. code-block:: bash

   $ basilisp run -c '(println *command-line-args*)' 1 2 3
   [1 2 3]
   $ basilisp run -c '(println *command-line-args*)'
   nil

.. _run_basilisp_applications:

Run Basilisp as an Application
------------------------------

Python applications don't have nearly as many constraints on their entrypoints as do Java applications.
Nevertheless, developers may have a clear entrypoint in mind when designing their application code.
In such cases, it may be desirable to take advantage of the computed Python :external:py:data:`sys.path` to invoke your entrypoint.
To do so, you can use the ``basilisp run -n`` flag to invoke an namespace directly:

.. code-block:: bash

   basilisp run -n package.core

When invoking your Basilisp code via namespace name, the specified namespace name will be bound to the var :lpy:var:`*main-ns*` as a symbol.
This allows you to gate code which should only be executed when this namespace is executed as an entrypoint, but would otherwise allow you to ``require`` the namespace normally.

.. code-block:: clojure

   (when (= *main-ns* 'package.core)
      (start-app))

This approximates the Python idiom of gating execution on import using ``if __name__ == "__main__":``.

This variant of ``basilisp run`` also permits users to provide command line arguments bound to :lpy:var:`*command-line-args*` as described above.

.. note::

   Only ``basilisp run -n`` binds the value of :lpy:var:`*main-ns*`.
   In all other cases, it will be ``nil``.

.. _run_basilisp_tests:

Run Basilisp Tests
------------------

If you installed the `PyTest <https://docs.pytest.org/en/7.0.x/>`_ extra, you can also execute your test suite using the Basilisp CLI.

.. code-block:: bash

   basilisp test

Because Basilisp defers all testing logic to PyTest, you can use any standard PyTest arguments and flags from this entrypoint.

.. _bootstrap_cli_command:

Bootstrap Python Installation
-----------------------------

For some installations, it may be desirable to have Basilisp readily importable whenever the Python interpreter is started.
You can enable that as described in :ref:`bootstrapping`:

.. code-block:: bash

   basilisp bootstrap

If you would like to remove the bootstrapped Basilisp from your installation, you can remove it:

.. code-block:: bash

   basilisp bootstrap --uninstall