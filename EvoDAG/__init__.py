# Copyright 2015 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evolving Directed Acyclic Graph (EvoDAG).

Public names:

* ``EvoDAGE``: ensemble of EvoDAG models (``fit``/``predict``).
* ``Model``: a fitted model, obtained with ``EvoDAG(...).fit(X, y).model()``.
* ``Ensemble``: the fitted ensemble behind ``EvoDAGE``.
* ``EvoDAG``, ``RGP`` and ``RootGP``: three names for the search engine
  (``EvoDAG.base.EvoDAG``).

The single-model wrapper with the scikit-learn-like interface is a different
class that is also called ``EvoDAG``: ``EvoDAG.model.EvoDAG``.

Names that need the optional dependency scikit-learn (``pip install
'EvoDAG[sklearn]'``) are loaded on first use and are not in ``__all__``:

* ``EvoDAGClassifier`` and ``EvoDAGRegressor``: scikit-learn estimators, each
  one fits a single model (``EvoDAG.estimators``).
* ``EnsembleClassifier`` and ``EnsembleRegressor``: scikit-learn
  meta-estimators that combine clones of any estimator with the rules of
  ``EvoDAGE`` (majority vote, median); ``EnsembleClassifier(EvoDAGClassifier())``
  gives the predictions of ``EvoDAGE``.
"""
from .base import EvoDAG, RGP, RootGP
from .model import EvoDAGE, Model, Ensemble
__all__ = ["EvoDAG", "RGP", "RootGP", "EvoDAGE", "Model", "Ensemble"]
__version__ = '0.18.0'


_ESTIMATORS = ("EvoDAGClassifier", "EvoDAGRegressor", "EnsembleClassifier",
               "EnsembleRegressor")


def __getattr__(name):
    # PEP 562: the estimators import scikit-learn, an optional dependency, so
    # they are imported on first access and not in ``__all__``.
    if name in _ESTIMATORS:
        from . import estimators
        return getattr(estimators, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
