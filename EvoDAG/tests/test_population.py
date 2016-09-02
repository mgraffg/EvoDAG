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


def test_generational_generation():
    from EvoDAG.population import Generational
    from EvoDAG import EvoDAG
    gp = EvoDAG(population_class='Generational',
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
                    all_inputs=True,
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
                all_inputs=True,
                popsize=3)
    gp.X = X
    gp.y = y
    gp.create_population()
    print(len(gp.population.population), len(gp.X))
    assert len(gp.population.population) == len(gp.X)
    for i in range(gp.popsize):
        a = gp.random_offspring()
        gp.replace(a)
    assert len(gp.population.population) == gp.popsize


def test_all_init_popsize():
    from EvoDAG import EvoDAG
    y = cl.copy()
    y[y != 1] = -1
    gp = EvoDAG(population_class='Generational',
                all_inputs=True,
                early_stopping_rounds=1,
                popsize=2)
    gp.X = X
    gp.y = y
    gp.create_population()
    assert gp.init_popsize == len(gp.X)
    gp = EvoDAG(population_class='Generational',
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
        gp = EvoDAG(population_class=pop,
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
                all_inputs=True,
                early_stopping_rounds=1,
                popsize=2)
    gp.X = X
    gp.y = y
    gp.create_population()
    for i in range(3):
        gp.replace(gp.random_offspring())
    assert gp.population.generation == 2
    

def test_clean():
    from EvoDAG import EvoDAG
    y = cl.copy()
    y[y != 1] = -1
    for pc in ['Generational', 'SteadyState']:
        gp = EvoDAG(population_class=pc,
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
