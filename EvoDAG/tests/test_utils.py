# Copyright 2016 Mario Graff Guerrero

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


def test_parameter_grid_results():
    from EvoDAG.utils import RandomParameterSearch
    a = RandomParameterSearch()
    assert len([x for x in a]) == 1468


def test_process_params():
    from EvoDAG.utils import RandomParameterSearch
    from EvoDAG import EvoDAG
    rs = RandomParameterSearch(npoints=1)
    args = [x for x in rs][0]
    evo = EvoDAG(**rs.process_params(args))
    params = evo.get_params()
    for k, v in args.items():
        if k in params:
            if k == 'generations':
                v = np.inf
            print(k, v, params[k])
            if isinstance(v, list):
                for a, b in zip(v, params[k]):
                    assert a == b.__name__
            elif hasattr(params[k], '__name__'):
                assert v == params[k].__name__
            else:
                assert v == params[k]


def test_random_parameter_search_len():
    from itertools import product
    from EvoDAG.utils import RandomParameterSearch
    params = {'a': [1, 2, 3],
              'b': [-1, -2],
              'c': ['a', 'b', 'c', 'd', 'e']}
    rs = RandomParameterSearch(params=params)
    params = sorted(params.items())
    a = product(*[x[1] for x in params])
    assert len([x for x in a]) == len(rs)


def test_random_parameter_search_getitem():
    from itertools import product
    from EvoDAG.utils import RandomParameterSearch
    params = {'a': [1, 2, 3],
              'b': [-1, -2],
              'c': ['a', 'b', 'c', 'd', 'e']}
    rs = RandomParameterSearch(params=params)
    params = sorted(params.items())
    a = product(*[x[1] for x in params])
    for k, v in enumerate(a):
        v1 = [x[1] for x in sorted(rs[k].items())]
        print(v, v1)
        assert np.all([x == y for x, y in zip(v, v1)])

    
def test_params():
    from EvoDAG.utils import RandomParameterSearch
    rs = RandomParameterSearch(npoints=734)
    l = np.unique([x['random_generations'] for x in rs])
    assert len(l) == 2
    l = np.unique([x['population_class'] for x in rs])
    assert len(l) == 2


def test_iter_params():
    from EvoDAG.utils import RandomParameterSearch
    params = dict(popsize=[100, 1000],
                  population_class=['SteadyState'],
                  early_stopping_rounds=[50, 1000])
    rs = RandomParameterSearch(params=params, npoints=5)
    args = [x for x in rs]
    assert len(args) == 4


def test_constraints():
    from EvoDAG.utils import RandomParameterSearch
    params = dict(popsize=[100, 1000],
                  population_class=['Generational',
                                    'SteadyState'],
                  early_stopping_rounds=[50, 1000])
    rs = RandomParameterSearch(params=params, npoints=32)
    args = [x for x in rs if x['population_class'] == 'Generational']
    assert len(args) == 2


def test_constraints2():
    from EvoDAG.utils import RandomParameterSearch
    params = dict(popsize=[100, 1000],
                  population_class=['Generational',
                                    'SteadyState'],
                  early_stopping_rounds=[100, 1000])
    rs = RandomParameterSearch(params=params, npoints=32, training_size=70)
    args = [x for x in rs if x['population_class'] == 'Generational']
    assert len(args) == 1


def test_popsize_constraint():
    from EvoDAG.utils import RandomParameterSearch
    params = dict(popsize=[100, 1000],
                  population_class=['Generational',
                                    'SteadyState'],
                  early_stopping_rounds=[100, 1000])
    rs = RandomParameterSearch(params=params, npoints=32, training_size=70)
    popsize = [x['popsize'] for x in rs]
    r = [x for x in popsize if x > 70]
    assert len(r) == 0


def test_inputs_vec():
    from test_root import X, cl
    from EvoDAG.utils import Inputs
    D = []
    for x, y in zip(X, cl):
        a = {}
        a['vec'] = x.tolist()
        a['klass'] = int(y)
        D.append(a)
    inputs = Inputs()
    X, y = inputs.read_data_json_vec(None, D)


def test_inputs_vecsize():
    from test_root import X, cl
    from EvoDAG.utils import Inputs
    D = []
    for x, y in zip(X, cl):
        a = {}
        a['vec'] = [[k, v] for k, v in enumerate(x)]
        a['klass'] = int(y)
        a['vecsize'] = len(x)
        D.append(a)
    inputs = Inputs()
    X, y = inputs.read_data_json_vec(None, D)


def test_inputs():
    from test_root import X, cl
    from EvoDAG.utils import Inputs
    D = []
    for x, y in zip(X, cl):
        a = {k: v for k, v in enumerate(x)}
        a['klass'] = int(y)
        a['num_terms'] = len(x)
        D.append(a)
    inputs = Inputs()
    X, y = inputs.read_data_json(None, D)
    
