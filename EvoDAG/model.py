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
from multiprocessing import Pool
import gc
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


def decision_function(model_X):
    model, X = model_X
    return model.decision_function(X)


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
    def multiple_outputs(self):
        return self._hist[0]._multiple_outputs

    @property
    def classifier(self):
        "whether this is classification or regression task"
        return self._classifier

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

    def decision_function(self, X, **kwargs):
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
            if self.multiple_outputs:
                r = [x.boundaries() for x in node.hy]
            else:
                r = node.hy.boundaries()
        else:
            r = node.hy
        for i in hist[:-1]:
            i.hy = None
            i.hy_test = None
        gc.collect()
        return r

    def predict(self, X, **kwargs):
        if self._classifier:
            if self.multiple_outputs:
                hy = self.decision_function(X, **kwargs)
                hy = np.array([x.tonparray() for x in hy])
                hy = hy.argmax(axis=0)
                if self._labels is not None:
                    hy = self._labels[hy]
            else:
                hy = self.decision_function(X, **kwargs).sign()
                if self._labels is not None:
                    hy = (hy + 1).sign()
                    hy = self._labels[hy.tonparray().astype(np.int)]
            return hy
        return self.decision_function(X, **kwargs).tonparray()

    def graphviz(self, fpt):
        flag = False
        if isinstance(fpt, str):
            flag = True
            fpt = open(fpt, 'w')
        fpt.write("digraph EvoDAG {\n")
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
            else:
                cdn = "n{0} [label=\"X{1}\" fillcolor=red style=filled];\n"
                fpt.write(cdn.format(k, n._variable))
        fpt.write("}\n")
        if flag:
            fpt.close()

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
    def classifier(self):
        "whether this is classification or regression task"
        return self.models[0].classifier

    @property
    def models(self):
        "List containing the different models. One model for each class"
        return self._models

    @property
    def fitness_vs(self):
        "Median Fitness in the validation set"
        l = [x.fitness_vs for x in self.models]
        return np.median(l)

    def __iter__(self):
        "Iterates on the models"
        for i in self.models:
            yield i

    def __len__(self):
        "Number of models"
        return len(self.models)

    def decision_function(self, X, **kwargs):
        return [x.decision_function(X) for x in self._models]

    def predict(self, X, **kwargs):
        d = self.decision_function(X, **kwargs)
        d = np.array([x.tonparray() for x in d])
        hy = d.argmax(axis=0)
        if self._labels is not None:
            hy = self._labels[hy]
        return hy

    def graphviz(self, skel):
        for k, m in enumerate(self.models):
            m.graphviz(skel + '-%s.gv' % k)


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
    def multiple_outputs(self):
        return self.models[0].multiple_outputs

    @property
    def classifier(self):
        return self._classifier

    def _decision_function_raw(self, X, cpu_cores=1):
        if cpu_cores == 1:
            r = [m.decision_function(X) for m in self._models]
        else:
            p = Pool(cpu_cores)
            args = [(m, X) for m in self._models]
            r = [x for x in tqdm(p.imap_unordered(decision_function,
                                                  args),
                                 total=len(args))]
            p.close()
        return r

    def decision_function(self, X, cpu_cores=1):
        if self.classifier:
            return self.decision_function_cl(X, cpu_cores=cpu_cores)
        r = self._decision_function_raw(X, cpu_cores=cpu_cores)
        if isinstance(r[0], SparseArray):
            r = np.array([x.tonparray() for x in r if x.isfinite()])
            sp = SparseArray.fromlist
            r = sp(np.median(r, axis=0))
        else:
            r = np.array([[y.tonparray() for y in x] for x in r])
            sp = SparseArray.fromlist
            r = np.median(r, axis=0)
            r = [sp(x) for x in r]
        return r

    def decision_function_cl(self, X, cpu_cores=1):
        r = self._decision_function_raw(X, cpu_cores=cpu_cores)
        res = r[0]
        if isinstance(res, SparseArray):
            res = res.boundaries()
        else:
            res = [x.boundaries() for x in res]
        for x in r[1:]:
            if isinstance(x, SparseArray):
                res = res + x.boundaries()
            else:
                res = [x + y.boundaries() for (x, y) in zip(res, x)]
        if isinstance(res, SparseArray):
            return res / len(r)
        else:
            return [x / len(r) for x in res]

    def predict(self, X, cpu_cores=1):
        if self.classifier:
            return self.predict_cl(X, cpu_cores=cpu_cores)
        return self.decision_function(X, cpu_cores=cpu_cores).tonparray()

    def predict_cl(self, X, cpu_cores=1):
        hy = self.decision_function(X, cpu_cores=cpu_cores)
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

    def graphviz(self, directory):
        "Directory to store the graphviz models"
        import os
        if not os.path.isdir(directory):
            os.mkdir(directory)
        output = os.path.join(directory, 'evodag-%s')
        for k, m in enumerate(self.models):
            m.graphviz(output % k)
