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


import numpy as np
from .sparse_array import SparseArray
from .node import Variable, Function


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
        self._hist = [self.transform(self._hist[x].tostore()) for x in
                      self._trace]
        self._labels = labels

    @property
    def fitness_vs(self):
        "Fitness in the validation set"
        return self._hist[-1].fitness_vs

    def transform(self, v):
        if not isinstance(v, Function):
            return v
        if v.nargs == 1:
            v.variable = self._map[v.variable]
        else:
            v.variable = [self._map[x] for x in v.variable]
        return v

    def decision_function(self, X):
        "Decision function i.e. the raw data of the prediction"
        if X is None:
            if self._classifier:
                return self._hy_test.boundaries()
            return self._hy_test
        X = self.convert_features(X)
        hist = self._hist
        for node in hist:
            if isinstance(node, Function):
                node.eval(hist)
            else:
                node.eval(X)
        if self._classifier:
            return node.hy.boundaries()
        return node.hy

    def predict(self, X):
        if self._classifier:
            hy = self.decision_function(X).sign()
            if self._labels is not None:
                hy = (hy + 1).sign()
                hy = self._labels[hy.tonparray().astype(np.int)]
            return hy
        return self.decision_function(X).tonparray()

    def graphviz(self, fpt):
        fpt.write("digraph RGP {\n")
        # fpt.write("""edge [dir="none"];\n""")
        # fpt.write("""edge;\n""")
        last = len(self._hist) - 1
        for k, n in enumerate(self._hist):
            if isinstance(n, Function):
                name = n.__class__.__name__
                extra = "colorscheme=blues9 style=filled color={0}".format(n.color)
                if k == last:
                    extra = "fillcolor=blue style=filled"
                fpt.write("n{0} [label=\"{1}\" {2}];\n".format(k,
                                                               name,
                                                               extra))
                vars = n._variable
                if not isinstance(vars, list):
                    vars = [vars]
                for j in vars:
                    fpt.write("n{0} -> n{1};\n".format(k, j))
                    # fpt.write("n{0} -> n{1};\n".format(j, k))
            else:
                cdn = "n{0} [label=\"X{1}\" fillcolor=red style=filled];\n"
                fpt.write(cdn.format(k, n._variable))
        fpt.write("}\n")

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

    @staticmethod
    def convert_features_test_set(vars, v):
        if isinstance(v, np.ndarray):
            X = v.T
        else:
            X = v
        for var, d in zip(vars, X):
            if isinstance(d, SparseArray):
                var._eval_ts = d
            else:
                var._eval_ts = SparseArray.fromlist(d)


class Models(object):
    "List of model in multiclass classification"
    def __init__(self, models, labels=None):
        self._models = models
        self._labels = labels

    @property
    def models(self):
        "List containing the different models. One model for each class"
        return self._models

    def __iter__(self):
        "Iterates on the models"
        for i in self.models:
            yield i

    def __len__(self):
        "Number of models"
        return len(self.models)

    def decision_function(self, X):
        return [x.decision_function(X) for x in self._models]

    def predict(self, X):
        d = self.decision_function(X)
        d = np.array([x.tonparray() for x in d])
        hy = d.argmax(axis=0)
        if self._labels is not None:
            hy = self._labels[hy]
        return hy


class Ensemble(object):
    "Ensemble that predicts using the average"
    def __init__(self, models):
        self._models = models
        self._labels = self._models[0]._labels
        self._classifier = False
        flag = False
        if isinstance(self._models[0], Models):
            if self._models[0]._models[0]._classifier:
                flag = True
        elif self._models[0]._classifier:
            flag = True
        self._classifier = flag

    @property
    def models(self):
        "List containing the models that compose the ensemble"
        return self._models

    @property
    def classifier(self):
        return self._classifier

    def decision_function(self, X):
        if self.classifier:
            return self.decision_function_cl(X)
        r = [m.decision_function(X) for m in self._models]
        r = np.array([x.tonparray() for x in r if x.isfinite()])
        sp = SparseArray.fromlist
        r = sp(np.median(r, axis=0))
        return r

    def decision_function_cl(self, X):
        r = [m.decision_function(X) for m in self._models]
        res = r[0]
        for x in r[1:]:
            if isinstance(x, SparseArray):
                res = res + x
            else:
                res = [x + y for (x, y) in zip(res, x)]
        if isinstance(res, SparseArray):
            return res / len(r)
        else:
            return [x / len(r) for x in res]

    def predict(self, X):
        if self.classifier:
            return self.predict_cl(X)
        return self.decision_function(X).tonparray()

    def predict_cl(self, X):
        hy = self.decision_function_cl(X)
        if isinstance(hy, SparseArray):
            hy = hy.sign()
            if self._labels is not None:
                hy = (hy + 1).sign()
                hy = self._labels[hy.tonparray().astype(np.int)]
        else:
            d = np.array([x.tonparray() for x in hy])
            hy = d.argmax(axis=0)
            if self._labels is not None:
                hy = self._labels[hy]
        return hy

