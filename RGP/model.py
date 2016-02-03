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

from .sparse_array import SparseArray
from .node import Variable, Function
import numpy as np


class Model(object):
    """Object to store the necesary elements to make predictions
    based on an individual"""
    def __init__(self, trace, hist, classifier=True, labels=None):
        self._classifier = classifier
        self._trace = trace
        self._hist = hist
        self._map = {}
        for k, v in enumerate(self._trace):
            self._map[v] = k
        self._hy_test = self._hist[self._trace[-1]].hy_test
        self._hist = map(lambda x: self.transform(self._hist[x].tostore()),
                         self._trace)
        self._labels = labels

    def transform(self, v):
        if not isinstance(v, Function):
            return v
        if v.nargs == 1:
            v.variable = self._map[v.variable]
        else:
            v.variable = map(lambda x: self._map[x], v.variable)
        return v

    def decision_function(self, X):
        "Decision function i.e. the raw data of the prediction"
        if X is None:
            return self._hy_test
        X = self.convert_features(X)
        hist = self._hist
        for node in hist:
            if isinstance(node, Function):
                node.eval(hist)
            else:
                node.eval(X)
        return node.hy

    def predict(self, X):
        if self._classifier:
            hy = self.decision_function(X).sign()
            if self._labels is not None:
                hy = (hy + 1).sign()
                hy = self._labels[hy.tonparray().astype(np.int)]
                hy = SparseArray.fromlist(hy)
            return hy
        return self.decision_function(X)

    @staticmethod
    def convert_features(v):
        if v is None:
            return None
        if isinstance(v[0], Variable):
            return v
        if isinstance(v, np.ndarray):
            X = v.T
        else:
            X = v
        lst = []
        for var, d in enumerate(X):
            v = Variable(var, 1)
            if isinstance(d, SparseArray):
                v._eval_tr = d
            else:
                v._eval_tr = SparseArray.fromlist(d)
            lst.append(v)
        return lst


class Models(object):
    "List of model in multiclass classification"
    def __init__(self, models, labels=None):
        self._models = models
        self._labels = labels

    def decision_function(self, X):
        return map(lambda x: x.decision_function(X), self._models)

    def predict(self, X):
        d = self.decision_function(X)
        d = np.array(map(lambda x: x.tonparray(), d))
        hy = d.argmax(axis=0)
        if self._labels is not None:
            hy = self._labels[hy]
        return SparseArray.fromlist(hy)
