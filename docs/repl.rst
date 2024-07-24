.. _repl:

REPL
====

The REPL is an interactive programming environment which Basilisp users can manipulate the runtime environment in real time, iteratively developing and changing their code one line at a time.
You can start up the REPL from a command-line using the command:

.. code-block:: shell

   basilisp repl

From there you'll be greeted with a prompt showing the current namespace followed by ``=>``:

.. code-block::

   basilisp.user=>

The default REPL namespace is ``basilisp.user``, but users can modify that at startup using the ``--default-ns`` flag to ``basilisp repl``.

.. note::

   REPL is an acronym meaning ``Read-Eval-Print-Loop`` which is refreshingly self-descriptive.

.. _repl_utilities:

REPL Utilities
--------------

Within the REPL you have access to the full suite of tools offered by Basilisp.
:lpy:ns:`basilisp.core` functions have been automatically referred into the default REPL namespace and are immediately available to you.

.. lpy:currentns:: basilisp.core

The following Vars are defined and may be useful when using the REPL:

* :lpy:var:`*e` holds the most recent exception that was thrown (if one)
* :lpy:var:`*1`, :lpy:var:`*2`, :lpy:var:`*3`: hold the value of the last 3 results

.. lpy:currentns:: basilisp.repl

Additionally, every public function from :lpy:ns:`basilisp.repl` is referred into the default REPL.
This namespace includes some utilities for introspecting the runtime environment including viewing source code and fetching docstrings.

* :lpy:fn:`doc` is a macro which takes a symbol as an argument and prints the documentation for a Basilisp Var
* :lpy:fn:`print-doc` is a function which prints the docstring for any Basilisp Var or Python object
* :lpy:fn:`source` is a macro which takes a symbol as an argument and prints the source code for a Basilisp Var
* :lpy:fn:`print-source` is a function which prints the source for any Basilisp Var or Python object

.. note::

   :lpy:fn:`doc` and :lpy:fn:`source` are macros which accept symbols and resolve to Vars before calling :lpy:fn:`print-doc` and :lpy:fn:`print-source` respectively.
   If you intend to call the latter functions with a Basilisp var, be sure to fetch the Var directly using :lpy:form:`var`.

   .. code-block::

      (doc map)
      (print-doc (var map))
      (source filter)
      (print-source (var filter))

.. lpy:currentns:: basilisp.core

.. seealso::

   :lpy:fn:`require`, :lpy:fn:`refer`, :lpy:fn:`use`

.. _repl_creature_comforts:

Creature Comforts
-----------------

Basilisp serves its REPL using the excellent Python `prompt-toolkit <https://github.com/prompt-toolkit/python-prompt-toolkit>`_ library, which enables a huge number of great usability features:

* Text completions for previously interned :ref:`keywords` (with and without namespaces) and any :ref:`vars` in scope in the current namespace
* File-backed REPL history (with shell-like history search)
* Multi-line input for incomplete forms
* :ref:`repl_syntax_highlighting`

.. note::

   You can configure where your REPL history file is stored by setting the ``BASILISP_REPL_HISTORY_FILE_PATH`` environment variable in your shell.
   By default it is stored in ``$XDG_DATA_HOME/basilisp/.basilisp_history``.

.. _repl_syntax_highlighting:

Syntax Highlighting
-------------------

Basilisp's command-line REPL can highlight your code using `Pygments <https://pygments.org/>`_ if the optional ``pygments`` extra is installed alongside Basilisp.
You can install it via Pip:

.. code-block:: shell

   pip install basilisp[pygments]

The default Pygments `style <https://pygments.org/styles/>`_ is ``emacs``, but you can select another style by setting the value of the ``BASILISP_REPL_PYGMENTS_STYLE_NAME`` environment variable in your shell.

.. note::

   If Pygments is installed, Basilisp will always display syntax highlighting in a shell context.
   To disable color output temporarily, you can set the ``BASILISP_NO_COLOR`` environment variable to ``true``.
