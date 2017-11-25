# Copyright 2013 Mario Graff Guerrero

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
import json
import os
import gzip


def line_iterator(filename):
    if filename.endswith(".gz"):
        f = gzip.GzipFile(filename)
    else:
        try:
            f = open(filename, encoding='utf8')
        except TypeError:
            f = open(filename)
    while True:
        line = f.readline()
        # Test the type of the line and encode it if neccesary...
        if type(line) is bytes:
            try:
                line = str(line, encoding='utf8')
            except TypeError:
                line = str(line)
        # If the line is empty, we are done...
        if len(line) == 0:
            break
        line = line.strip()
        # If line is empty, jump to next...
        if len(line) == 0:
            continue
        yield line
    # Close the file...
    f.close()


def json_iterator(filename):
    for line in line_iterator(filename):
        yield json.loads(line)


def tonparray(a):
    return np.array(a.full_array())


def BER(y, yh):
    u = np.unique(y)
    b = 0
    for cl in u:
        m = y == cl
        b += (~(y[m] == yh[m])).sum() / float(m.sum())
    return (b / float(u.shape[0])) * 100.


def RSE(x, y):
    return ((x - y)**2).sum() / ((x - x.mean())**2).sum()


params_fname = os.path.join(os.path.dirname(__file__), 'conf', 'parameter_values.json')
with open(params_fname, 'r') as fpt:
    PARAMS = json.loads(fpt.read())


class Inputs(object):
    def __init__(self):
        self._word2id = {}
        self._label2id = {}

    @property
    def word2id(self):
        return self._word2id

    @property
    def label2id(self):
        return self._label2id

    @word2id.setter
    def word2id(self, s):
        self._word2id = s

    @label2id.setter
    def label2id(self, s):
        self._label2id = s

    @staticmethod
    def _num_terms(a):
        if 'num_terms' in a:
            num_terms = a['num_terms']
        else:
            num_terms = len(a)
            if 'klass' in a:
                num_terms -= 1
        return num_terms

    def convert(self, x):
        try:
            return float(x)
        except ValueError:
            if x not in self.word2id:
                self.word2id[x] = len(self.word2id)
            return self.word2id[x]

    def convert_label(self, x):
        try:
            x = float(x)
            if np.isfinite(x):
                return x
            x = str(x)
        except ValueError:
            pass
        if x not in self.label2id:
            self.label2id[x] = len(self.label2id)
        return self.label2id[x]

    def _read_csv(self, fname):
        X = []
        for i in line_iterator(fname):
            x = i.rstrip().lstrip()
            if len(x):
                X.append([i for i in x.split(',')])
        return X

    def read_csv(self, fname, dim):
        X = []
        y = []
        d = self._read_csv(fname)
        if dim > 1:
            for x in d:
                X.append([self.convert(i) for i in x[:-dim]])
                y.append(x[-dim:])
            X = np.array(X)
            y = [SparseArray.fromlist([float(x[i]) for x in y]) for i in range(dim)]
        elif dim == 1:
            for x in d:
                X.append([self.convert(i) for i in x[:-1]])
                y.append(self.convert_label(x[-1]))
            X = np.array(X)
            y = np.array(y)
        else:
            X, y = np.array([[self.convert(i) for i in x] for x in d]), None
        return X, y

    def read_data_json(self, fname, iterable=None):
        X = None
        y = []
        dependent = os.getenv('KLASS')
        if dependent is None:
            dependent = 'klass'
        if iterable is None:
            iterable = json_iterator(fname)
        for row, a in enumerate(iterable):
            if 'vec' in a:
                return self.read_data_json_vec(fname)
            if X is None:
                X = [list() for i in range(self._num_terms(a))]
            for k, v in a.items():
                try:
                    k = int(k)
                    X[k].append((row, self.convert(v)))
                except ValueError:
                    if k == dependent:
                        y.append(self.convert_label(v))
        num_rows = row + 1
        X = [SparseArray.index_data(x, num_rows) for x in X]
        if len(y) == 0:
            y = None
        else:
            y = np.array(y)
        return X, y

    def read_data_json_vec(self, fname, iterable=None):
        X = None
        y = []
        dependent = os.getenv('KLASS')
        if dependent is None:
            dependent = 'klass'
        if iterable is None:
            iterable = json_iterator(fname)
        for row, a in enumerate(iterable):
            vec = a['vec']
            try:
                vecsize = a['vecsize']
            except KeyError:
                vecsize = len(vec)
                vec = enumerate(vec)
            if X is None:
                X = [list() for i in range(vecsize)]
            for k, v in vec:
                k = int(k)
                X[k].append((row, self.convert(v)))
            try:
                y.append(self.convert_label(a[dependent]))
            except KeyError:
                pass
        num_rows = row + 1
        X = [SparseArray.index_data(x, num_rows) for x in X]
        if len(y) == 0:
            y = None
        else:
            y = np.array(y)
        return X, y


class RandomParameterSearch(object):
    def __init__(self, params=PARAMS,
                 npoints=1468,
                 training_size=5000,
                 seed=0):
        self._training_size = training_size
        self.popsize_constraint(params)
        self._params = sorted(params.items())
        assert len(self._params)
        self._params.reverse()
        self._len = None
        self._npoints = npoints
        self._seed = seed
        self.fix_early_popsize()

    def popsize_constraint(self, params):
        try:
            params['popsize'] = [x for x in params['popsize'] if x <= self._training_size]
        except KeyError:
            pass

    def fix_early_popsize(self):
        try:
            popsize = [x for x in self._params if x[0] == 'popsize'][0]
            if len(popsize[1]) == 0:
                popsize[1].append(self._training_size)
        except IndexError:
            pass
        try:
            early = [x for x in self._params if x[0] == 'early_stopping_rounds'][0]
            early_min = min(early[1])
            if early_min > self._training_size:
                early[1].append(self._training_size)
        except IndexError:
            pass

    def __len__(self):
        if self._len is None:
            _ = np.array([len(x[1]) for x in self._params], dtype=np.uint)
            _ = np.product(_)
            assert _ >= 0
            self._len = _
        return self._len

    def __getitem__(self, key):
        res = {}
        lens = [len(x[1]) for x in self._params]
        for l, k_v in zip(lens, self._params):
            k, v = k_v
            key, residual = divmod(key, l)
            res[k] = v[int(residual)]
        return res

    def constraints(self, k):
        try:
            if k['population_class'] == 'Generational' and\
               k['early_stopping_rounds'] < k['popsize']:
                return False
            if k['early_stopping_rounds'] > self._training_size:
                return False
        except KeyError:
            return True
        return True

    def __iter__(self):
        np.random.seed(self._seed)
        m = {}
        _len = self.__len__()
        npoints = self._npoints if _len > self._npoints else _len
        while npoints:
            k = np.round(np.random.uniform(0, _len)).astype(np.uint)
            if len(m) == _len:
                return
            while k in m:
                k = np.random.randint(_len)
            m[k] = 1
            p = self[k]
            if self.constraints(p):
                npoints -= 1
                yield p

    @staticmethod
    def process_params(a):
        from EvoDAG import EvoDAG
        fs_class = {}
        function_set = []
        for x in EvoDAG()._function_set:
            fs_class[x.__name__] = x
        args = {}
        for k, v in a.items():
            if k in fs_class:
                if not isinstance(v, bool):
                    fs_class[k].nargs = v
                if v:
                    function_set.append(fs_class[k])
            else:
                args[k] = v
            fs_evo = EvoDAG()._function_set
            fs_evo = filter(lambda x: x in function_set, fs_evo)
            args['function_set'] = [x for x in fs_evo]
        return args
