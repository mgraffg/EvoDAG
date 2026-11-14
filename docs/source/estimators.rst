The scikit-learn estimators
===========================

:py:mod:`EvoDAG.estimators`
---------------------------

:py:class:`~EvoDAG.estimators.EvoDAGClassifier` and
:py:class:`~EvoDAG.estimators.EvoDAGRegressor` give one EvoDAG model the
interface of scikit-learn (``get_params``, ``clone``, pipelines, grid search,
``score``). Each instance is one evolution and one
:py:class:`~EvoDAG.model.Model`, and its predictions equal those of
``EvoDAG.model.EvoDAG`` with the same parameters and seed. The module needs
scikit-learn, an optional dependency (``pip install 'EvoDAG[sklearn]'``).

.. automodule:: EvoDAG.estimators
   :members:
   :exclude-members: set_score_request
