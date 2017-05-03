# Copyright 2017 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from EvoDAG.function_selection import FunctionSelection
from nose.tools import assert_almost_equals


def test_seed():
    a = FunctionSelection(seed=0, nfunctions=10)
    rf = a.random_function()
    flag = False
    for seed in range(1, 10):
        b = FunctionSelection(seed=seed, nfunctions=10)
        if b.random_function() != rf:
            flag = True
    assert flag
    for seed in range(1, 10):
        b = FunctionSelection(seed=0, nfunctions=10)
        if b.random_function() != rf:
            assert False


def test_setitem():
    b = FunctionSelection(seed=0, nfunctions=10)
    assert b.fitness[3] == 0
    assert b.times[3] == 0
    b[3] = 3.2
    assert b.fitness[3] == 3.2
    assert b.times[3] == 1


def test_avg_fitness():
    b = FunctionSelection(seed=0, nfunctions=10)
    b[3] = 3.2
    b[3] = 5
    print(b.avg_fitness(3), (3.2 + 5) / 2.)
    assert b.avg_fitness(3) == (3.2 + 5) / 2.


def test_tournament():
    b = FunctionSelection(seed=0, nfunctions=10)
    for i in range(100):
        c = b.tournament()
        b[c] = -3.2
    for i in range(10):
        print(b.avg_fitness(i), -3.2)
        assert_almost_equals(b.avg_fitness(i), -3.2)


def test_number_args():
    for s in range(10):
        a = FunctionSelection(seed=s, nfunctions=2, nargs=[1, 2])
        index = a.tournament()
        assert index == 1


def test_density():
    a = FunctionSelection(seed=0, nfunctions=3, nargs=[1, 2, 2], density_safe=[1, 2])
    assert a.min_density == 0 and a.density == 1.0
    a.min_density = 0.9
    a.density = 0.7
    for i in range(10):
        index = a.tournament()
        assert index == 1 or index == 2
    a = FunctionSelection(seed=0, nfunctions=2, nargs=[1, 2], density_safe=[1])
    assert a.min_density == 0 and a.density == 1.0
    a.min_density = 0.9
    a.density = 0.7
    index = a.tournament()
    assert index == 1


def test_unfeasible_functions():
    a = FunctionSelection(seed=0, nfunctions=3, nargs=[1, 2, 2], density_safe=[1, 2])
    assert len(a.unfeasible_functions) == 0
    a.unfeasible_functions.add(1)
    a.unfeasible_functions.add(0)
    x = a.tournament()
    assert x == 2
    a = FunctionSelection(seed=0, nfunctions=3, nargs=[1, 2, 2])
    assert len(a.unfeasible_functions) == 0
    a.unfeasible_functions.add(1)
    a.unfeasible_functions.add(0)
    x = a.tournament()
    assert x == 2
    a.unfeasible_functions.add(2)
    x = a.tournament()
