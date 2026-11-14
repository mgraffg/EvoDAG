Compiled modules and legacy code
================================

The numerical kernels are written in Cython; they are compiled when the
package is installed (``python setup.py build_ext --inplace`` in a checkout).
Their functions work on arrays of ``float64``.

:py:mod:`EvoDAG.array_utils`
----------------------------

Operations over one-dimensional arrays: elementwise operations that keep
the conventions of the evolution (a zero stays zero), reductions, and
counts.

.. automodule:: EvoDAG.array_utils
   :members:

:py:mod:`EvoDAG.cython_utils`
-----------------------------

Naive Bayes, centroids and the classification scores used as fitness.

.. automodule:: EvoDAG.cython_utils
   :members:

:py:mod:`EvoDAG.linalg_solve`
-----------------------------

The least-squares solver that computes the weights of a node.

.. automodule:: EvoDAG.linalg_solve
   :members:

:py:mod:`EvoDAG.gp`
-------------------

Individuals written in prefix notation; this module is not used by the
engine.

.. automodule:: EvoDAG.gp
   :members:
