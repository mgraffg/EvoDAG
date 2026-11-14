Nodes and weights
=================

:py:mod:`EvoDAG.node`
---------------------

The individuals. :py:class:`~EvoDAG.node.Variable` is an input variable
scaled by a coefficient and :py:class:`~EvoDAG.node.Function` is the base of
the functions that combine other individuals. The coefficients of a node are
found by ordinary least squares
(:py:meth:`~EvoDAG.node.Variable.compute_weight`), except those of the Naive
Bayes and Centroid nodes, which come from the class labels.

.. automodule:: EvoDAG.node
   :members:
   :show-inheritance:

:py:mod:`EvoDAG.naive_bayes`
----------------------------

.. automodule:: EvoDAG.naive_bayes
   :members:
