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


def BER(y, yh):
    u = np.unique(y)
    b = 0
    for cl in u:
        m = y == cl
        b += (~(y[m] == yh[m])).sum() / float(m.sum())
    return (b / float(u.shape[0])) * 100.


def RSE(x, y):
    return ((x - y)**2).sum() / ((x - x.mean())**2).sum()


class RandomParameterSearch(object):
    def __init__(self, params={'Add': [2, 5, 10, 15, 20, 25, 30],
                               'Mul': [0, 2, 5, 10, 15, 20, 25, 30],
                               'Min': [0, 2, 5, 10, 15, 20, 25, 30],
                               'Max': [0, 2, 5, 10, 15, 20, 25, 30],
                               'Div': [True, False],
                               'Fabs': [True, False],
                               'Exp': [True, False],
                               'Sqrt': [True, False],
                               'Sin': [True, False],
                               'Cos': [True, False],
                               'Ln': [True, False],
                               'Sq': [True, False],
                               'Sigmoid': [True, False],
                               'If': [True, False],
                               'unique_individuals': [True, False],
                               'popsize': [500, 1000, 2000, 3000],
                               'early_stopping_rounds':
                               [125, 250, 500, 1000, 2000]},
                 npoints=1468,
                 seed=0):
        self._params = sorted(params.items())
        self._params.reverse()
        self._len = None
        self._npoints = npoints
        self._seed = seed

    def __len__(self):
        if self._len is None:
            _ = np.product([len(x[1]) for x in self._params])
            self._len = _
        return self._len

    def __getitem__(self, key):
        res = {}
        lens = [len(x[1]) for x in self._params]
        for l, k_v in zip(lens, self._params):
            k, v = k_v
            key, residual = divmod(key, l)
            res[k] = v[residual]
        return res

    def __iter__(self):
        np.random.seed(self._seed)
        m = {}
        for i in range(self._npoints):
            k = np.random.randint(len(self))
            while k in m:
                k = np.random.randint(len(self))
            m[k] = 1
            yield self[k]

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
