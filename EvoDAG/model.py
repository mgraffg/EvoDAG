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
from SparseArray import SparseArray
from .node import Variable, Function
from .utils import tonparray
from multiprocessing import Pool
import logging
import gc
import os
import gzip
import pickle
import shutil
from time import time
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x
LOGGER = logging.getLogger('EvoDAG')


def fit(X_y_evodag):
    X, y, test_set, evodag, tmpdir, init_time = X_y_evodag
    if tmpdir is not None:
        seed = evodag['seed']
        output = os.path.join(tmpdir, '%s.evodag' % seed)
        if os.path.isfile(output):
            with gzip.open(output) as fpt:
                try:
                    return pickle.load(fpt)
                except Exception:
                    pass
    try:
        time_limit = evodag['time_limit']
        if time_limit is not None:
            evodag['time_limit'] = time_limit - (time() - init_time)
            if evodag['time_limit'] < 2:
                LOGGER.info('Not enough time (seed: %s) ' % evodag['seed'])
                return None
    except KeyError:
        pass
    try:
        evodag = EvoDAG(**evodag)
        evodag.fit(X, y, test_set=test_set)
    except RuntimeError:
        return None
    m = evodag.model
    gc.collect()
    if tmpdir is not None:
        with gzip.open(output, 'w') as fpt:
            pickle.dump(m, fpt)
    return m


def decision_function(model_X):
    k, model, X = model_X
    return [k, model.decision_function(X)]


def predict_proba(model_X):
    k, model, X = model_X
    return [k, model.predict_proba(X)]


class Model(object):
    """Object to store the necesary elements to make predictions
    based on an individual"""
    def __init__(self, trace, hist, nvar=None, classifier=True, labels=None,
                 probability_calibration=None, nclasses=None):
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
        self._nvar = nvar
        self._probability_calibration = probability_calibration
        self._nclasses = nclasses
        
    @property
    def nclasses(self):
        return self._nclasses

    @property
    def nvar(self):
        return self._nvar

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

    @property
    def size(self):
        return len(self._hist)

    @property
    def height(self):
        return self._hist[-1].height

    def inputs(self, counter=None):
        from collections import Counter
        if counter is None:
            counter = Counter()
        for node in self._hist:
            if node.height == 0:
                if isinstance(node._variable, list):
                    for _ in node._variable:
                        counter[_] += 1
                else:
                    counter[node._variable] += 1
        return counter

    def transform(self, v):
        if v.height == 0:
            return v
        if v.nargs == 1:
            v.variable = self._map[v.variable]
        else:
            v.variable = [self._map[x] for x in v.variable]
        return v

    def predict_proba(self, X, **kwargs):
        X = self.decision_function(X, **kwargs)
        return self._probability_calibration.predict_proba(X)

    def decision_function(self, X, **kwargs):
        "Decision function i.e. the raw data of the prediction"
        if X is None:
            return self._hy_test
        X = self.convert_features(X)
        if len(X) < self.nvar:
            _ = 'Number of variables differ, trained with %s given %s' % (self.nvar, len(X))
            raise RuntimeError(_)
        hist = self._hist
        for node in hist:
            if node.height:
                node.eval(hist)
            else:
                node.eval(X)
        node.normalize()
        r = node.hy
        for i in hist[:-1]:
            i.hy = None
            i.hy_test = None
        gc.collect()
        return r

    def predict(self, X, **kwargs):
        hy = self.decision_function(X, **kwargs)
        if self._classifier:
            [x.finite(inplace=True) for x in hy]
            hy = np.array(SparseArray.argmax(hy).full_array(), dtype=np.int)
            if self._labels is not None:
                hy = self._labels[hy]
        else:
            hy = tonparray(hy)
        return hy

    def graphviz(self, fpt, terminals=True):
        flag = False
        if isinstance(fpt, str):
            flag = True
            fpt = open(fpt, 'w')
        fpt.write("digraph EvoDAG {\n")
        last = len(self._hist) - 1
        height = self._hist[-1].height
        try:
            b, m = np.linalg.solve([[0, height-1], [1, 1]], [9, 1])
        except np.linalg.linalg.LinAlgError:
            b, m = 0, 1
        done = {}
        for k, n in enumerate(self._hist):
            if isinstance(n, Function):
                done[k] = 1
                name = n.__class__.__name__
                if n.height == 0:
                    cdn = "n{0} [label=\"{1}\" fillcolor=red style=filled];\n"
                    fpt.write(cdn.format(k, name))
                    continue
                color = int(np.round(n.height * m + b))
                extra = "colorscheme=blues9 style=filled color={0}".format(color)
                if k == last:
                    extra = "fillcolor=green style=filled"
                fpt.write("n{0} [label=\"{1}\" {2}];\n".format(k,
                                                               name,
                                                               extra))
                vars = n._variable
                if not isinstance(vars, list):
                    vars = [vars]
                for j in vars:
                    if j in done:
                        fpt.write("n{0} -> n{1};\n".format(k, j))
            elif terminals:
                cdn = "n{0} [label=\"X{1}\" fillcolor=red style=filled];\n"
                fpt.write(cdn.format(k, n._variable))
                done[k] = 1
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
        elif isinstance(v[0], SparseArray):
            X = v
        else:
            X = np.array(v).T
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


class Ensemble(object):
    "Ensemble that predicts using the average"
    def __init__(self, models, n_jobs=1, evodags=None, tmpdir=None):
        self._models = models
        self._n_jobs = n_jobs
        self._evodags = evodags
        self._tmpdir = tmpdir
        if models is not None:
            self._init()

    def fit(self, X, y, test_set=None):
        evodags = self._evodags
        init_time = time()
        args = [(X, y, test_set, evodag, self._tmpdir, init_time) for evodag in evodags]
        try:
            time_limit = evodags[0]['time_limit']
        except KeyError:
            time_limit = None
        if time_limit is not None:
            LOGGER.info('time_limit in Ensemble: %0.2f' % time_limit)
        if self._n_jobs == 1:
            _ = [fit(x) for x in tqdm(args)]
            self._models = [x for x in _ if x is not None]
        else:
            p = Pool(self._n_jobs, maxtasksperchild=1)
            self._models = []
            for x in tqdm(p.imap_unordered(fit, args),
                          total=len(args)):
                if x is not None:
                    self._models.append(x)
                if time_limit is not None and time() - init_time > time_limit:
                    p.terminate()
                    break
            p.close()
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir)
        self._init()
        if time_limit is not None:
            LOGGER.info('Used time in Ensemble: %0.2f' % (time() - init_time))
        return self

    def _init(self):
        self._labels = self._models[0]._labels
        self._classifier = False
        flag = False
        if self._models[0]._classifier:
            flag = True
        self._classifier = flag

    @property
    def nclasses(self):
        return self.models[0].nclasses

    @property
    def probability_calibration(self):
        return self.models[0]._probability_calibration is not None

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

    @property
    def fitness_vs(self):
        "Median Fitness in the validation set"
        l = [x.fitness_vs for x in self.models]
        return np.median(l)

    @property
    def size(self):
        l = [x.size for x in self.models]
        return np.median(l)

    @property
    def height(self):
        l = [x.height for x in self.models]
        return np.median(l)

    def inputs(self, counter=None):
        from collections import Counter
        if counter is None:
            counter = Counter()
        for m in self.models:
            m.inputs(counter=counter)
        return counter

    def _decision_function_raw(self, X, cpu_cores=1):
        if cpu_cores == 1:
            r = [m.decision_function(X) for m in self._models]
        else:
            p = Pool(cpu_cores, maxtasksperchild=1)
            args = [(k, m, X) for k, m in enumerate(self._models)]
            r = [x for x in tqdm(p.imap_unordered(decision_function,
                                                  args),
                                 total=len(args))]
            r.sort(key=lambda x: x[0])
            r = [x[1] for x in r]
            p.close()
        return r

    def raw_decision_function(self, X):
        hy = self._decision_function_raw(X, cpu_cores=self._n_jobs)
        if isinstance(hy[0], list):
            _ = []
            [[_.append(y) for y in x] for x in hy]
            hy = _
        if self.classifier:
            [x.finite(inplace=True) for x in hy]
        return np.array([tonparray(x) for x in hy]).T

    def _predict_proba_raw(self, X, cpu_cores=1):
        if cpu_cores == 1:
            r = [m.predict_proba(X) for m in self._models]
        else:
            p = Pool(cpu_cores, maxtasksperchild=1)
            args = [(k, m, X) for k, m in enumerate(self._models)]
            r = [x for x in tqdm(p.imap_unordered(predict_proba,
                                                  args),
                                 total=len(args))]
            r.sort(key=lambda x: x[0])
            r = [x[1] for x in r]
            p.close()
        return r

    def predict_proba(self, X):
        if self.probability_calibration:
            proba = np.array(self._predict_proba_raw(X, cpu_cores=self._n_jobs))
            proba = np.mean(proba, axis=0)
            proba /= np.sum(proba, axis=1)[:, np.newaxis]
            proba[np.isnan(proba)] = 1. / self.nclasses
            return proba
        hy = self._decision_function_raw(X, cpu_cores=self._n_jobs)
        minlength = len(hy[0])
        hy = [SparseArray.argmax(x) for x in hy]
        hy = np.array([x.full_array() for x in hy], dtype=np.int).T
        hy = [np.bincount(x, minlength=minlength) for x in hy]
        return np.array([x / float(x.sum()) for x in hy])

    def decision_function(self, X, cpu_cores=1):
        cpu_cores = max(cpu_cores, self._n_jobs)
        r = self._decision_function_raw(X, cpu_cores=cpu_cores)
        if isinstance(r[0], SparseArray):
            r = np.array([tonparray(x) for x in r if x.isfinite()])
            r = np.median(r, axis=0)
        else:
            [[x.finite(inplace=True) for x in o] for o in r]
            r = np.array([[tonparray(y) for y in x] for x in r])
            r = np.median(r, axis=0)
        return r.T

    def predict(self, X, cpu_cores=1):
        cpu_cores = max(cpu_cores, self._n_jobs)
        if self.classifier:
            return self.predict_cl(X, cpu_cores=cpu_cores)
        return self.decision_function(X, cpu_cores=cpu_cores)

    def predict_cl(self, X, cpu_cores=1):
        cpu_cores = max(cpu_cores, self._n_jobs)
        hy = [SparseArray.argmax(x) for x in
              self._decision_function_raw(X, cpu_cores=cpu_cores)]
        hy = np.array([x.full_array() for x in hy], dtype=np.int).T
        hy = [np.bincount(x).argmax() for x in hy]
        if self._labels is not None:
            hy = self._labels[hy]
        return hy

    def graphviz(self, directory, **kwargs):
        "Directory to store the graphviz models"
        import os
        if not os.path.isdir(directory):
            os.mkdir(directory)
        output = os.path.join(directory, 'evodag-%s')
        for k, m in enumerate(self.models):
            m.graphviz(output % k, **kwargs)

    @classmethod
    def init(cls, n_estimators=30, n_jobs=1, tmpdir=None, **kwargs):
        try:
            init_seed = kwargs['seed']
            del kwargs['seed']
        except KeyError:
            init_seed = 0
        lst = []
        for x in range(init_seed, init_seed + n_estimators):
            kwargs['seed'] = x
            lst.append(kwargs.copy())
        if tmpdir is not None and not os.path.isdir(tmpdir):
            os.mkdir(tmpdir)
        return cls(None, evodags=lst, n_jobs=n_jobs, tmpdir=tmpdir)


class EvoDAGE(object):
    def __init__(self, time_limit=None, **kwargs):
        self._m = Ensemble.init(time_limit=time_limit, **kwargs)
        self._time_limit = time_limit

    @property
    def time_limit(self):
        return self._time_limit

    @time_limit.setter
    def time_limit(self, time_limit):
        self._time_limit = time_limit
        for x in self._m._evodags:
            x['time_limit'] = self._time_limit

    def fit(self, *args, **kwargs):
        return self._m.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self._m.predict(*args, **kwargs)

    def decision_function(self, *args, **kwargs):
        return self._m.decision_function(*args, **kwargs)

    def raw_decision_function(self, *args, **kwargs):
        return self._m.raw_decision_function(*args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        return self._m.predict_proba(*args, **kwargs)


class EvoDAG(EvoDAGE):
    def __init__(self, **kwargs):
        from EvoDAG import EvoDAG as evodag
        self._m = evodag.init(**kwargs)

    def fit(self, *args, **kwargs):
        self._m.fit(*args, **kwargs)
        self._m = self._m.model()
        return self

    @property
    def model(self):
        return self._m
