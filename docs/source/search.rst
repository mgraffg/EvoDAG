Data, parameter search and command line
=======================================

:py:mod:`EvoDAG.utils`
----------------------

Error measures, the readers of the CSV and JSON files and the random search
of parameters.

.. automodule:: EvoDAG.utils
   :members:

:py:mod:`EvoDAG.command_line`
-----------------------------

The four console commands, ``EvoDAG-params``, ``EvoDAG-train``,
``EvoDAG-predict`` and ``EvoDAG-utils``, are the functions
:py:func:`~EvoDAG.command_line.params`, :py:func:`~EvoDAG.command_line.train`,
:py:func:`~EvoDAG.command_line.predict` and
:py:func:`~EvoDAG.command_line.utils`. They read files and ``sys.argv``, so
they have no doctest: each docstring names the tests in
``tests/test_command_line.py`` that verify it. The options are listed in the
Command line page of the user site.

.. automodule:: EvoDAG.command_line
   :members:
   :show-inheritance:
