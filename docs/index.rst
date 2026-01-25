.. Basilisp documentation master file, created by
   sphinx-quickstart on Fri Sep 14 08:39:59 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Basilisp's documentation!
====================================

.. image:: https://img.shields.io/badge/Github-basilisp-green?style=flat-square
   :target: https://github.com/basilisp-lang/basilisp
   :alt: Link to Basilisp Github repository
.. image:: https://img.shields.io/pypi/v/basilisp.svg?style=flat-square
   :target: https://pypi.org/project/basilisp/
   :alt: Link to Basilisp PyPI page for current release; shows the current release version
.. image:: https://img.shields.io/pypi/pyversions/basilisp.svg?style=flat-square
   :target: https://pypi.org/project/basilisp/
   :alt: Link to Basilisp PyPI page for current release; shows currently supported Python versions
.. image:: https://img.shields.io/readthedocs/basilisp.svg?style=flat-square
   :target: https://docs.basilisp.org
   :alt: Link to Basilisp documentation
.. image:: https://github.com/basilisp-lang/basilisp/actions/workflows/run-tests.yml/badge.svg?branch=main&style=flat-square
   :target: https://github.com/basilisp-lang/basilisp/actions/workflows/run-tests.yml
   :alt: Link to Basilisp test CI workflow on Github Actions
.. image:: https://github.com/basilisp-lang/basilisp/actions/workflows/run-clojure-test-suite.yml/badge.svg?branch=main&style=flat-square
   :target: https://github.com/basilisp-lang/basilisp/actions/workflows/run-clojure-test-suite.yml
   :alt: Link to Basilisp clojure-test-suite CI workflow on Github Actions
.. image:: https://img.shields.io/coveralls/github/basilisp-lang/basilisp.svg?style=flat-square
   :target: https://coveralls.io/github/basilisp-lang/basilisp
   :alt: Link to Basilisp Coverage report on Coveralls
.. image:: https://img.shields.io/github/license/basilisp-lang/basilisp.svg?style=flat-square
   :target: https://github.com/basilisp-lang/basilisp/blob/master/LICENSE
   :alt: Link to Basilisp license file
.. image:: https://img.shields.io/badge/Slack-Clojurians-green?style=flat-square
   :target: https://clojurians.slack.com/archives/C071RFV2Z1D
   :alt: Link to Basilisp channel on Clojurians Slack

Basilisp is a :ref:`Clojure-compatible(-ish) <differences_from_clojure>` Lisp dialect hosted on Python 3 with seamless Python interop.

Basilisp compiles down to raw Python 3 code and executes on the Python 3 virtual machine, allowing natural interoperability between existing Python libraries and new Lisp code.

Use the links below to learn more about Basilisp and to find help guide you as you are using Basilisp.

.. note::

   This documentation strives to be correct and complete, but if you do find a issue, please feel free to `file an issue on GitHub <https://github.com/basilisp-lang/basilisp/issues>`_.

Contents
--------

.. toctree::
   :maxdepth: 2

   features
   gettingstarted
   differencesfromclojure
   concepts
   reference
   releasenotes
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`lpy-nsindex`
* :ref:`search`