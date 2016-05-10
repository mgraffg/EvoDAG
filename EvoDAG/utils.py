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

PARAMS = {'Add': [2, 5, 10, 15, 20, 25, 30],
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
          'early_stopping_rounds': [125, 250, 500, 1000, 2000]}


def BER(y, yh):
    u = np.unique(y)
    b = 0
    for cl in u:
        m = y == cl
        b += (~(y[m] == yh[m])).sum() / float(m.sum())
    return (b / float(u.shape[0])) * 100.


def RSE(x, y):
    return ((x - y)**2).sum() / ((x - x.mean())**2).sum()


def parameter_grid_results(npoints=1468):
    from sklearn import grid_search
    from EvoDAG import EvoDAG
    fs = EvoDAG().function_set
    ps = grid_search.ParameterSampler(PARAMS, n_iter=npoints, random_state=0)
    args = []
    for m in ps:
        x1 = m.copy()
        function_set = [x for x in fs if m.get(x.__name__, True)]
        if fs[0] not in function_set:
            function_set.insert(0, fs[0])
            x1['Add'] = 2
        for x in fs:
            x = x.__name__
            if isinstance(x1[x], bool):
                del x1[x]
        x1['function_set'] = function_set
        args.append(x1)
    return args


def process_params(a):
    from EvoDAG import EvoDAG
    fs_class = {}
    for x in EvoDAG()._function_set:
        fs_class[x.__name__] = x
    args = {}
    for k, v in a.items():
        if k in fs_class:
            fs_class[k].nargs = v
        else:
            args[k] = v
    return args
