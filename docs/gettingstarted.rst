.. _getting_started:

Getting Started
===============

.. _installation_and_first_steps:

Installation & First Steps
--------------------------

Basilisp is developed on `GitHub <https://github.com/chrisrink10/basilisp>`_ and hosted on `PyPI <https://pypi.python.org/pypi/basilisp>`_.
You can fetch Basilisp using a simple::

    pip install basilisp

Once Basilisp is installed, you can enter into the REPL using::

    basilisp repl

In Basilisp's REPL, you now have the full power of Basilisp at your disposal.
It is customary to write a ``Hello, World!`` when starting out in a new language, so we'll do that here::

    basilisp.user=> (print "Hello, World!")
    Hello, World!
    nil

Or perhaps you'd like to try something a little more exciting, like performing some arithmetic::

    basilisp.user=> (+ 1 2 3 4 5)
    15

Sequences are a little more fun than simple arithmetic::

    basilisp.user=> (filter odd? (map inc (range 1 10)))
    (3 5 7 9 11)

There is a ton of great functionality built in to Basilisp, so feel free to poke around.
Many great features from Clojure are already baked right in, and `many more are planned <https://github.com/chrisrink10/basilisp/issues>`_, so I hope you enjoy.
From here you might find the documentation for the :ref:`repl` helpful to learn about what else you can do in the REPL.

Good luck!

.. _using_basilisp_in_a_project:

Using Basilisp in a Project
---------------------------

.. _project_structure:

Project Structure
^^^^^^^^^^^^^^^^^

Basilisp projects are broadly structured the same as their Python counterparts, albeit with more strict structure than what Python actually allows.
Source code should be placed under a ``src`` directory and tests under ``tests``, both directly in the project root.
By convention, the "main" namespace of a Basilisp project is called ``core``, though that is not required.
Sub or child namespaces may be nested using folders.
A namespace may be both a "leaf" and "branch" node in the source tree without any special configuration, as ``myproject.pkg`` is below.
Basilisp source files should always have a ``.lpy`` extension.

::

   .
   ├── README.md
   ├── poetry.lock
   ├── pyproject.toml
   ├── src
   │   └── myproject
   │       ├── core.lpy
   │       ├── pkg
   │       │   └── subns.lpy
   │       └── pkg.lpy
   └── tests
       └── myproject
           └── test_core.lpy

.. note::

   Python ``__init__.py`` files are not required anywhere in Basilisp projects (including for nested namespaces), though you may need to use them if your project mixes Python and Basilisp sources.

Basilisp apps can use any of Python's myriad dependency management options, including `pip <https://pip.pypa.io/en/stable/>`_, `Pipenv <https://pipenv.pypa.io/en/latest/>`_, and `Poetry <https://python-poetry.org/>`_.
Basilisp itself uses Poetry and that is the recommended

.. _bootstrapping:

Bootstrapping
^^^^^^^^^^^^^

The REPL can certainly be a useful tool for exploration and iterative development, but if you plan to use Basilisp in a project you'll need to bootstrap it at the entrypoint of your project.
Bootstrapping is the process of initializing the Basilisp runtime, setting up the :lpy:ns:`basilisp.core` namespace, and preparing the runtime to evaluate and execute your code.
Without bootstrapping, you would be unable to run any Basilisp code!

Basilisp includes a couple of different bootstrapping functions depending on how you intend to use it.

For tools with a clear entrypoint, such as a CLI tool, you can trivially wrap your project's entrypoint written in Basilisp code with a simple Python wrapper by simply calling out to the :py:func:`basilisp.main.bootstrap` function.
Given a Basilisp entrypoint function ``main`` (taking no arguments) in the ``project.core`` namespace, you can have Basilisp bootstrap itself and then call your function directly.

.. code-block:: python

   from basilisp.main import bootstrap


   def invoke_cli():
        bootstrap("project.core:main")

If you were to place this in a module such as ``myproject.main``, you could easily configure a `setuptools entry point <https://setuptools.pypa.io/en/latest/userguide/entry_point.html>`_ (or any analog with another build tool) to point to that script directly, effectively launching you directly to Basilisp code.

For more sophisticated projects which may not have a direct or wrappable entrypoint, you can initialize the Basilisp runtime directly by calling :py:func:`basilisp.main.init` with no arguments.
This may be a better fit for a project using something like Django, where the entrypoint is dictated by Django.
In that case, you could use a hook such as Django's ``AppConfig.ready()``.

.. code-block:: python

   import basilisp.main
   from django.apps import AppConfig


   class MyAppConfig(AppConfig):
       def ready(self):
           basilisp.main.init()

Any Basilisp namespace can be imported directly and run once :py:func:`basilisp.main.init` has run.
Basilisp code will operate normally (calling into other Basilisp namespaces and functions) after initialization is completed.

.. note::

   Manual bootstrapping is designed to be as simple as possible, but it is not the long term goal of this project's maintainers that it should be necessary.
   Eventually, we plan to release a tool akin to Python's Poetry, or similar tools in other languages that helps facilitate both dependency management and packaging in such a way that bootstrapping is completely transparent to the developer.
