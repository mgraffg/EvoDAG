The search engine
=================

``EvoDAG.base`` holds the engine class that runs one evolution. It creates
the offspring (choosing a function, selecting its arguments, computing the
weights), asks the population to replace an individual with it and decides
when to stop.

:py:mod:`EvoDAG.base`
---------------------

.. automodule:: EvoDAG.base
   :members:
   :show-inheritance:

:py:mod:`EvoDAG.population`
---------------------------

The population stores the individuals, runs the tournaments (selection of
the arguments and negative selection of the individual that leaves) and
tracks the best individual on the validation split.

.. automodule:: EvoDAG.population
   :members:
   :show-inheritance:

:py:mod:`EvoDAG.function_selection`
-----------------------------------

The tournament that chooses the function of each offspring according to the
average fitness the function has obtained (compiled module).

.. automodule:: EvoDAG.function_selection
   :members:
