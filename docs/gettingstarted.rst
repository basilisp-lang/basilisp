Getting Started
===============

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