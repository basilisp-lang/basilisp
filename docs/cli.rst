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

An nREPL server provides an interactive REPL environment for remote code execution and development from an editor.

Basilisp's CLI incorporates an nREPL server adapted from `nbb <https://github.com/babashka/nbb>`_.

Start from an editor with a Clojure extension supporting Basilisp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

   The extension's ``Jack-in`` command is used to start an nREPL session and connect your editor to it.

With Emacs `CIDER v1.14 <https://docs.cider.mx/cider/platforms/basilisp.html>`_ and Visual Studio Code `Calva v2.0.453 <https://calva.io/basilisp/>`_ or later, you can ``Jack-in`` to a Basilisp project directly.
The extensions also recognize Basilisp ``.lpy`` files as Clojure files.

To ``Jack-in`` to a Basilisp project

1. Ensure that a ``basilisp.edn`` file is present at the root of your project, even if it is empty.
2. Run the ``Jack-in`` command in your editor and select ``Basilisp`` if prompted.
   The Editor should then start the server and connect to it.

Start from an editor with a Clojure extension not yet supporting Basilisp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::

   The extension's ``connect`` command is used to connect your editor to a running nREPL server.
   It requires the port number where the server is running at.

.. note::

   Basilisp code files use the ``.lpy`` filename suffix.
   You might need to adjust your editor's extension settings to recognize this suffix as a Clojure file.
   Alternatively, you can evaluate code inside ``.clj`` files, though importing these files from other Basilisp files might not be possible due to the different file extension.

If your editor extension does not yet support Basilisp, or if you prefer more control over the nREPL server, you can start the server from the command line and connect to using your extension's ``connect`` command.

The nREPL server when started will provide the host and port number it is listening on.

To view available command line options, use

.. code-block:: bash

   basilisp nrepl-server -h


To start the server on a random port, use

.. code-block:: bash

   basilisp nrepl-server
   # => nREPL server started on port 50407 on host 127.0.0.1 - nrepl://127.0.0.1:50407


To start the server on a specific port, use

.. code-block:: bash

   basilisp nrepl-server --port 8889
   #=> nREPL server started on port 8889 on host 127.0.0.1 - nrepl://127.0.0.1:8889

Some extensions can connect to a running server automatically by looking for a ``.nrepl-port`` file at the root of the project. This file contains the port the server is listening on.

The ``nrepl-server`` command will generate this file in the current working directory where the server is started.
If your extension's ``connect`` command looks for this file, run the server command from the root of the project, so that is generated in there

.. code-block:: bash

   cd <project-root-directory>
   basilisp nrepl-server
   #=> nREPL server started on port 632128 on host 127.0.0.1 - nrepl://127.0.0.1:63128

Alternatively, specify the full path where this file should be generated using the ``--port-filepath`` CLI option

.. code-block:: bash

   basilisp nrepl-server --port-filepath <project-root-directory>/.nrepl-port
   #=> nREPL server started on port 62079 on host 127.0.0.1 - nrepl://127.0.0.1:62079

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
