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


from test_root import X, cl
from nose.tools import assert_almost_equals
from test_command_line import default_nargs


def test_generational_generation():
    from EvoDAG.population import Generational
    from EvoDAG import EvoDAG
    function_set = [x for x in EvoDAG()._function_set if x.regression]
    gp = EvoDAG(population_class='Generational', classifier=False,
                function_set=function_set,
                popsize=10)
    gp.X = X
    y = cl.copy()
    y[y != 1] = -1
    gp.y = y
    gp.create_population()
    assert isinstance(gp.population, Generational)
    p = []
    for i in range(gp.popsize-1):
        a = gp.random_offspring()
        p.append(a)
        gp.replace(a)
    assert len(gp.population._inner) == (gp.popsize - 1)
    a = gp.random_offspring()
    p.append(a)
    gp.replace(a)
    assert len(gp.population._inner) == 0
    for a, b in zip(gp.population.population, p):
        assert a == b


def test_all_inputs():
    from EvoDAG import EvoDAG
    y = cl.copy()
    y[y != 1] = -1
    for pc in ['Generational', 'SteadyState']:
        gp = EvoDAG(population_class=pc,
                    all_inputs=True, classifier=False,
                    popsize=10)
        gp.X = X
        gp.y = y
        gp.create_population()
        assert len(gp.population.population) < 10
        for i in range(gp.population.popsize,
                       gp.population._popsize):
            a = gp.random_offspring()
            gp.replace(a)
        assert len(gp.population.population) == 10


def test_all_inputs2():
    from EvoDAG import EvoDAG
    y = cl.copy()
    y[y != 1] = -1
    gp = EvoDAG(population_class='Generational',
                all_inputs=True, classifier=False,
                pr_variable=1, popsize=3)
    gp.X = X
    gp.y = y
    gp.create_population()
    print(len(gp.population.population), len(gp.X))
    assert len(gp.population.population) == len(gp.X)
    for i in range(gp.popsize):
        a = gp.random_offspring()
        gp.replace(a)
    assert len(gp.population.population) == gp.popsize


def test_all_inputs3():
    from EvoDAG import EvoDAG
    y = cl.copy()
    y[y != 1] = -1
    gp = EvoDAG(population_class='SteadyState',
                all_inputs=True, classifier=False,
                pr_variable=1, popsize=3)
    gp.X = X
    gp.y = y
    gp.create_population()
    print(len(gp.population.population), len(gp.X))
    assert len(gp.population.population) == gp.population.popsize
    for i in range(gp.popsize):
        a = gp.random_offspring()
        gp.replace(a)
    assert len(gp.population.population) == gp.popsize
    

def test_all_init_popsize():
    from EvoDAG import EvoDAG
    y = cl.copy()
    y[y != 1] = -1
    gp = EvoDAG(population_class='Generational',
                all_inputs=True, classifier=False,
                early_stopping_rounds=1, pr_variable=1, popsize=2)
    gp.X = X
    gp.y = y
    gp.create_population()
    assert gp.init_popsize == len(gp.X)
    gp = EvoDAG(population_class='Generational', classifier=False,
                # all_inputs=True,
                early_stopping_rounds=1,
                popsize=2)
    gp.X = X
    gp.y = y
    gp.create_population()
    assert gp.init_popsize == gp.popsize


def test_random_generations():
    from EvoDAG import EvoDAG
    from EvoDAG.population import SteadyState

    class P(SteadyState):
        def random_selection(self, negative=False):
            raise RuntimeError('!')
    y = cl.copy()
    y[y != 1] = -1
    for pop in ['SteadyState', 'Generational', P]:
        gp = EvoDAG(population_class=pop, classifier=False,
                    all_inputs=True, random_generations=1,
                    early_stopping_rounds=1, popsize=2)
        gp.X = X
        gp.y = y
        gp.create_population()
        print(gp.population._random_generations)
        assert gp.population._random_generations == 1
        if pop == P:
            try:
                ind = gp.random_offspring()
                gp.replace(ind)
                assert False
            except RuntimeError:
                pass
        else:
            for i in range(3):
                gp.replace(gp.random_offspring())
            assert gp.population.generation == 2


def test_SteadyState_generation():
    from EvoDAG import EvoDAG
    y = cl.copy()
    y[y != 1] = -1
    gp = EvoDAG(population_class='SteadyState',
                all_inputs=True, classifier=False,
                early_stopping_rounds=1,
                popsize=2)
    gp.X = X
    gp.y = y
    gp.create_population()
    for i in range(3):
        gp.replace(gp.random_offspring())
    assert gp.population.generation == 2
    

def test_clean():
    from EvoDAG.node import Centroid
    from EvoDAG import EvoDAG
    Centroid.nargs = 0
    y = cl.copy()
    y[y != 1] = -1
    for pc in ['Generational', 'SteadyState']:
        gp = EvoDAG(population_class=pc, classifier=False,
                    popsize=5)
        gp.X = X
        gp.y = y
        gp.create_population()
        for i in range(10):
            v = gp.random_offspring()
            gp.replace(v)
        pop = gp.population.population
        esi = gp.population.estopping
        for i in gp.population._hist:
            print(i == esi, i in pop, i, '-'*10, i.fitness)
            if i == esi:
                assert i.hy is not None
            elif i in pop:
                assert i.hy is not None
        assert gp.population.estopping.hy is not None
    Centroid.nargs = 2


def test_density():
    from EvoDAG import EvoDAG
    Xc = X.copy()
    Xc[0, 0] = 0
    Xc[1, 1] = 0
    y = cl.copy()
    y[y != 1] = -1
    for pc in ['Generational', 'SteadyState']:
        gp = EvoDAG(population_class=pc, classifier=False,
                    popsize=5)
        gp.X = Xc
        gp.y = y
        gp.create_population()
        d = sum([x.hy.density for x in gp.population.population]) / gp.popsize
        print(d, gp.population.density, 'pop')
        assert gp.population.density == d
        for _ in range(3):
            a = gp.random_offspring()
            print(a.hy.density)
            gp.replace(a)
            d = sum([x.hy.density for x in gp.population.population]) / gp.popsize
            print(d, gp.population.density, 'replace')
            print(gp.population.density, d, '==')
            assert_almost_equals(gp.population.density, d)


def test_share_inputs():
    from EvoDAG import EvoDAG
    y = cl.copy()
    gp = EvoDAG(classifier=True, multiple_outputs=True,
                popsize=5, share_inputs=True)
    gp.fit(X, y)
    assert gp._share_inputs


def test_model_nvar():
    from EvoDAG import EvoDAG
    y = cl.copy()
    gp = EvoDAG(classifier=True, multiple_outputs=True,
                popsize=5, share_inputs=True)
    gp.fit(X, y)
    assert gp._share_inputs
    m = gp.model()
    print(X.shape)
    assert m.nvar == X.shape[1]
    try:
        m.predict(X[:, :3])
        assert False
    except RuntimeError:
        pass


def test_selectNumbers():
    from EvoDAG.cython_utils import SelectNumbers
    s = SelectNumbers([x for x in range(100)])
    a = s.get(10)
    print(a, len(a))
    assert len(a) == 10
    a = s.get(91)
    assert len(a) == 90
    s.pos -= 10
    a = s.get(10)
    assert len(a) == 10
    assert s.pos == 100
    assert s.empty()
    s.pos = 0
    while not s.empty():
        s.get_one()
    s = SelectNumbers([])
    assert s.empty()


def test_inputs():
    from EvoDAG.population import Inputs
    from EvoDAG.cython_utils import SelectNumbers
    from EvoDAG import EvoDAG
    y = cl.copy()
    gp = EvoDAG(classifier=True, multiple_outputs=True,
                popsize=5, share_inputs=True)
    gp.X = X
    gp.nclasses(y)
    gp.y = y
    inputs = Inputs(gp, SelectNumbers([x for x in range(len(gp.X))]))
    func = inputs._func
    for f in func:
        inputs._func = [f]
        inputs._nfunc = 1
        v = inputs.input()
        assert v is not None
        inputs = Inputs(gp, SelectNumbers([x for x in range(len(gp.X))]))


def test_multiple_variables():
    import numpy as np
    from EvoDAG.population import Inputs
    from EvoDAG.cython_utils import SelectNumbers
    from EvoDAG import EvoDAG
    from SparseArray import SparseArray
    y = cl.copy()
    gp = EvoDAG(classifier=True, multiple_outputs=True,
                popsize=5, share_inputs=True)
    gp.X = X
    gp.X[-1]._eval_tr = SparseArray.fromlist([0 for x in range(gp.X[-1].hy.size())])
    gp.nclasses(y)
    gp.y = y
    inputs = Inputs(gp, SelectNumbers([x for x in range(len(gp.X))]))
    inputs._func = [inputs._func[-1]]
    inputs._nfunc = 1
    v = inputs.input()
    assert v is not None
    mask = np.array(gp._mask[0].full_array(), dtype=np.bool)
    D = np.array([x.hy.full_array() for x in gp.X]).T
    b = np.array(gp._ytr[0].full_array())
    coef = np.linalg.lstsq(D[mask], b[mask])[0]
    for a, b in zip(coef, v.weight[0]):
        assert_almost_equals(a, b)


def test_inputs_func_argument():
    from EvoDAG import EvoDAG

    class Error:
        nargs = 2
        min_nargs = 2
        classification = True
        regression = True

        def __init__(self, *args, **kwargs):
            raise RuntimeError('aqui')
    y = cl.copy()
    gp = EvoDAG(classifier=True, multiple_outputs=True,
                pr_variable=0, input_functions=[Error], popsize=5, share_inputs=True)
    gp.X = X
    gp.nclasses(y)
    gp.y = y
    try:
        gp.create_population()
        assert False
    except RuntimeError:
        pass
    gp = EvoDAG(classifier=True, multiple_outputs=True,
                pr_variable=0, input_functions=['NaiveBayes', 'NaiveBayesMN',
                                                'MultipleVariables'],
                popsize=5, share_inputs=True).fit(X, y)
    assert gp
    try:
        EvoDAG(classifier=True, multiple_outputs=True,
               pr_variable=0, input_functions=['NaiveBayesXX', 'NaiveBayesMN',
                                               'MultipleVariables'],
               popsize=5, share_inputs=True).fit(X, y)
    except AttributeError:
        pass


def test_restrictions():
    from EvoDAG.population import Inputs
    from EvoDAG.cython_utils import SelectNumbers
    from EvoDAG import EvoDAG
    from SparseArray import SparseArray
    y = cl.copy()
    gp = EvoDAG(classifier=True, multiple_outputs=True,
                popsize=5, share_inputs=True)
    gp.X = X
    gp.X[-1]._eval_tr = SparseArray.fromlist([0 for x in range(gp.X[-1].hy.size())])
    gp.nclasses(y)
    gp.y = y
    for c in [True, False]:
        gp._classifier = c
        tag = 'classification' if c else 'regression'
        inputs = Inputs(gp, SelectNumbers([x for x in range(len(gp.X))]))
        for x in inputs._funcs:
            print(x, tag)
            assert getattr(x, tag)


def test_inputs_func_argument_regression():
    from EvoDAG import EvoDAG

    class Error:
        nargs = 2
        min_nargs = 2
        classification = True
        regression = True

        def __init__(self, *args, **kwargs):
            raise RuntimeError('aqui')
    y = cl.copy()
    y[y == 0] = -1
    y[y > -1] = 1
    gp = EvoDAG(classifier=False, multiple_outputs=False,
                pr_variable=0, input_functions=[Error],
                popsize=5, share_inputs=True)
    gp.X = X
    gp.nclasses(y)
    gp.y = y
    try:
        gp.create_population()
        assert False
    except RuntimeError:
        pass


def get_remote_data():
    import os
    import subprocess
    if not os.path.isfile('evodag.params'):
        subprocess.call(['curl', '-O', 'http://ws.ingeotec.mx/~mgraffg/evodag_data/evodag.params'])
    if not os.path.isfile('train.sp'):
        subprocess.call(['curl', '-O', 'http://ws.ingeotec.mx/~mgraffg/evodag_data/train.sp'])


def test_HGeneration():
    import json
    import gzip
    import pickle
    from EvoDAG.utils import RandomParameterSearch
    from EvoDAG import EvoDAG

    get_remote_data()
    params = json.loads(open('evodag.params').read())
    try:
        with gzip.open('train.sp') as fpt:
            X = pickle.load(fpt)
            y = pickle.load(fpt)
    except ValueError:
        return
    params['population_class'] = 'HGenerational'
    params['pr_variable'] = 0.1
    kw = RandomParameterSearch.process_params(params)
    gp = EvoDAG(**kw).fit(X, y)
    assert gp


def test_HGeneration_pr_variable():
    import json
    from EvoDAG.utils import RandomParameterSearch
    from EvoDAG import EvoDAG
    get_remote_data()
    params = json.loads(open('evodag.params').read())
    params['population_class'] = 'HGenerational'
    params['pr_variable'] = 1.0
    kw = RandomParameterSearch.process_params(params)
    y = cl.copy()
    try:
        EvoDAG(**kw).fit(X, y)
    except AssertionError:
        return
    assert False


def test_all_variables_inputs():
    from EvoDAG.population import Inputs
    from EvoDAG.cython_utils import SelectNumbers
    from EvoDAG import EvoDAG
    y = cl.copy()
    gp = EvoDAG(classifier=True, multiple_outputs=True,
                use_all_vars_input_functions=True,
                popsize=5, share_inputs=True)
    gp.X = X
    gp.nclasses(y)
    gp.y = y
    inputs = Inputs(gp, SelectNumbers([x for x in range(len(gp.X))]))
    func = inputs._func
    print(func, gp.nvar)
    for f in func:
        v = inputs.all_variables()
        assert v is not None
        assert isinstance(v, f)
    assert inputs._all_variables_index == len(func)


def test_input_functions():
    from EvoDAG import EvoDAG
    y = cl.copy()
    gp = EvoDAG.init(input_functions=["NaiveBayes", "NaiveBayesMN", "Centroid"],
                     Centroid=2, time_limit=5)
    input_functions = [x for x in gp._input_functions]
    gp.fit(X, y)
    default_nargs()
    for a, b in zip(input_functions, gp.population.hist[:3]):
        print(b, a)
        assert isinstance(b, a)


def test_popsize_nvar():
    from EvoDAG import EvoDAG
    y = cl.copy()
    gp = EvoDAG.init(popsize='nvar', time_limit=5)
    print(X.shape)
    gp.fit(X, y)
    default_nargs()
    assert gp.population._popsize == (X.shape[1] + len(gp._input_functions))
