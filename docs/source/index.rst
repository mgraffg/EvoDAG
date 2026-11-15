:mod:`EvoDAG`
=============

.. Registers the top-level package as a cross-reference target (it renders
   nothing): every module below is documented by its own ``automodule``, but
   the package itself is only referred to.
.. py:module:: EvoDAG

Version |version|.

.. note::

   This is the developer reference: the public API with examples that are
   executed and verified, and a one-line description of every other public
   name. Using EvoDAG (installation, the command line, the parameters, how
   the method works) is documented on the user site,
   `mgraffg.github.io/EvoDAG <https://mgraffg.github.io/EvoDAG/>`_.

EvoDAG is a steady-state semantic genetic programming system for
classification and regression. The usual way to use it is through the
scikit-learn estimators of :py:mod:`EvoDAG.estimators`:
``EnsembleClassifier(EvoDAGClassifier())`` (``EnsembleRegressor`` and
``EvoDAGRegressor`` for regression) is EvoDAG's ensemble, and
``EvoDAGClassifier`` alone fits one model. They need scikit-learn, an optional
dependency (``pip install 'EvoDAG[sklearn]'``). EvoDAG's own classes do not
need it and are exported by the top-level package: ``EvoDAGE`` (the ensemble),
``Model`` and ``Ensemble`` (what a fit returns), and ``EvoDAG``, ``RGP`` and
``RootGP`` (the three names of the search engine). Two different classes are
called ``EvoDAG``: ``EvoDAG.EvoDAG`` is the search engine of
:py:mod:`EvoDAG.base`, whereas ``EvoDAG.model.EvoDAG`` is a single model with
the ``fit``/``predict`` interface; each docstring says which one it is.

The first page is the scikit-learn estimators; the rest follow the way the
data flows through the package:

#. :doc:`estimators`: the scikit-learn estimators, one model or an ensemble
   (:py:mod:`EvoDAG.estimators`).
#. :doc:`engine`: the search engine (:py:mod:`EvoDAG.base`), the population
   and the selection of parents (:py:mod:`EvoDAG.population`,
   :py:mod:`EvoDAG.function_selection`).
#. :doc:`nodes`: the individuals and the functions, with the least-squares
   weights (:py:mod:`EvoDAG.node`, :py:mod:`EvoDAG.naive_bayes`).
#. :doc:`fitness`: the fitness, the training and validation masks
   (:py:mod:`EvoDAG.bagging_fitness`).
#. :doc:`model`: the fitted model and the ensemble (:py:mod:`EvoDAG.model`).
#. :doc:`search`: reading the data files, the random search of parameters and
   the command line (:py:mod:`EvoDAG.utils`, :py:mod:`EvoDAG.command_line`).
#. :doc:`extensions`: the compiled Cython modules and the legacy
   :py:mod:`EvoDAG.gp`.

Architecture
------------

The following example runs the engine on a tiny problem and looks at what it
leaves behind; it is executed when the documentation is tested.

>>> import numpy as np
>>> from EvoDAG import RGP
>>> rng = np.random.RandomState(0)
>>> X = rng.randn(100, 3)
>>> y = (X[:, 0] > 0).astype(int)
>>> engine = RGP.init(popsize=50, early_stopping_rounds=50, seed=0)
>>> engine = engine.fit(X, y)

**An individual is a node.** Every individual that the evolution creates is
an instance of a class of :py:mod:`EvoDAG.node`
(:py:class:`~EvoDAG.node.Variable` for an input variable, a subclass of
:py:class:`~EvoDAG.node.Function` for the rest) and is appended to
``engine.population.hist``, the *history*. A node stores in ``variable`` the
positions, in that list, of the nodes it was computed from (the index of the
column of ``X`` for an input variable), its ``weight`` (the coefficients),
its ``height`` and its fitness. As several nodes can point at one node, the
set of individuals is a directed acyclic graph. In the example the first
node of the history is the one that combines the three input variables, so
its ``variable`` lists the three columns.

>>> hist = engine.population.hist
>>> hist[0].height
0
>>> hist[0].variable
[0, 1, 2]

**Semantics: hy and hy_test.** A node holds the values it produces on the
training examples in ``hy`` and, if a test set was given to ``fit``, on the
test examples in ``hy_test`` (``None`` otherwise). In classification a node
has one array per class. The values are computed when the node is created,
from the ``hy`` of its arguments, so creating an offspring only costs
operations over arrays of the size of the data.

>>> best = engine.population.estopping
>>> len(best.hy), best.hy[0].shape
(2, (100,))
>>> best.hy_test is None
True

**Weights and fitness.** When an offspring is created, its coefficients are
computed by ordinary least squares against the target on the examples of the
training mask (:py:meth:`~EvoDAG.node.Variable.compute_weight`), and the
fitness is computed on the training part and on the validation part
(:py:class:`~EvoDAG.bagging_fitness.BaggingFitness`). The masks are made by
:py:mod:`EvoDAG.bagging_fitness` from ``tr_fraction`` and the seed.

**From the population to a model.** The engine keeps the *early stopping
individual*, the one with the best fitness on the validation part.
:py:meth:`EvoDAG.base.EvoDAG.model` collects the nodes it depends on
(:py:meth:`EvoDAG.base.EvoDAG.trace`), renumbers their positions and builds a
:py:class:`~EvoDAG.model.Model` with only those nodes and their weights.
The model evaluates its nodes in order on new inputs and does not keep the
training data.

>>> model = engine.model()
>>> model.size <= len(hist)
True
>>> model.nvar
3

**The ensemble.** :py:class:`~EvoDAG.model.EvoDAGE` runs the engine
``n_estimators`` times with consecutive seeds; each run has its own split of
the data, so the models differ. :py:class:`~EvoDAG.model.Ensemble`
aggregates them: the class that most models predict, the median of the
outputs for ``decision_function``, and the fraction of votes for
``predict_proba``.

.. toctree::
   :hidden:

   estimators
   engine
   nodes
   fitness
   model
   search
   extensions
