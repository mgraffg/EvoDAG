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
import logging
import numpy as np
from .node import Function, NaiveBayes, NaiveBayesMN, MultipleVariables
from .model import Model
from .cython_utils import SelectNumbers
import gc


class Inputs(object):
    def __init__(self, base, vars, functions=None):
        self._base = base
        self._vars = vars
        self._unique_individuals = set()
        if functions is None:
            self._funcs = [NaiveBayes, NaiveBayesMN, MultipleVariables]
        else:
            self._funcs = functions
        assert len(self._funcs) <= 3
        c = base._classifier
        tag = 'classification' if c else 'regression'
        self._funcs = [x for x in self._funcs if getattr(x, tag)]
        self.functions()

    def functions(self):
        base = self._base
        density = sum([x.hy.density for x in base.X]) / base.nvar
        func = [x for x in self._funcs if x.nargs > 0]
        if not len(func):
            self._func = None
            self._nfunc = 0
            return
        if density < self._base._min_density:
            func = [x for x in func if x.density_safe]
        self._nfunc = len(func)
        self._func = func

    def function(self):
        if self._nfunc == 1:
            return self._func
        elif self._nfunc == 2:
            if np.random.random() < 0.5:
                return self._func
            return [self._func[1], self._func[0]]
        else:
            rnd = np.random.random()
            if rnd < 0.3:
                return self._func
            elif rnd < 0.6:
                return [self._func[1], self._func[0], self._func[2]]
            else:
                return [self._func[2], self._func[0], self._func[1]]

    def input(self):
        base = self._base
        unique_individuals = self._unique_individuals
        vars = self._vars
        if self._nfunc == 0:
            return None
        for _ in range(base._number_tries_feasible_ind):
            args = []
            func = self.function()
            for f in func:
                nargs = f.nargs
                if len(args):
                    vars.pos -= len(args)
                if vars.empty():
                    return None
                args = vars.get(nargs)
                if len(args) < f.min_nargs:
                    return None
                v = f(args,
                      ytr=base._ytr, naive_bayes=base.naive_bayes,
                      finite=base._finite, mask=base._mask)
                sig = v.signature()
                if sig in unique_individuals:
                    continue
                unique_individuals.add(sig)
                v.height = 0
                if not v.eval(base.X):
                    continue
                if not v.isfinite():
                    continue
                if not base._bagging_fitness.set_fitness(v):
                    continue
                return v
        return None


class BasePopulation(object):
    def __init__(self, base=None,
                 tournament_size=2,
                 classifier=True,
                 labels=None,
                 popsize=10000,
                 random_generations=0,
                 es_extra_test=lambda x: True):
        self._base = base
        self._p = []
        self._hist = []
        self._bsf = None
        self._estopping = None
        self._tournament_size = tournament_size
        self._index = None
        self._classifier = classifier
        self._es_extra_test = es_extra_test
        self._labels = labels
        self._logger = logging.getLogger('EvoDAG')
        self._previous_estopping = False
        self._current_popsize = 0
        self._popsize = popsize
        self._inds_replace = 0
        self.generation = 1
        self._random_generations = random_generations
        self._density = 0.0

    @property
    def popsize(self):
        return self._current_popsize

    @property
    def population(self):
        "List containing the population"
        return self._p

    @population.setter
    def population(self, a):
        self._current_popsize = len(a)
        self._p = a

    @property
    def density(self):
        return self._density / self.popsize

    def get_density(self, v):
        try:
            return v.hy.density
        except AttributeError:
            return sum([x.density for x in v.hy]) / len(v.hy)

    @property
    def previous_estopping(self):
        """
        Returns whether the last individual set in the population was
        an early stopping individual
        """
        return self._previous_estopping

    @previous_estopping.setter
    def previous_estopping(self, v):
        self._previous_estopping = v

    @property
    def hist(self):
        "List containing all the individuals generated"
        return self._hist

    @property
    def bsf(self):
        "Best so far"
        return self._bsf

    @property
    def estopping(self):
        "Early stopping individual"
        return self._estopping

    @estopping.setter
    def estopping(self, v):
        self.previous_estopping = False
        if v.fitness_vs is None:
            return None
        flag = False
        if self.estopping is None:
            if not self._es_extra_test(v):
                return None
            self._estopping = v
            flag = True
        elif v.fitness_vs > self.estopping.fitness_vs:
            if not self._es_extra_test(v):
                return None
            self._estopping = v
            flag = True
        if flag:
            self.previous_estopping = flag
            self._base._unfeasible_counter = 0
            vfvs = v.fitness_vs
            self._logger.info('(%i) ES: %0.4f %0.4f' % (v.position,
                                                        v.fitness,
                                                        vfvs))

    @bsf.setter
    def bsf(self, v):
        flag = False
        if self.bsf is None:
            self._bsf = v
            flag = True
        elif v.fitness > self.bsf.fitness:
            self._bsf = v
            flag = True
        if flag:
            if v.fitness_vs is None:
                fvs = ""
            else:
                fvs = "%0.4f" % v.fitness_vs
            fts = "%0.4f" % v.fitness
            self._logger.debug('(%(position)s) BSF: %(fts)s %(fvs)s',
                               {'fts': fts, 'fvs': fvs, 'position': v.position})

    def model(self, v=None):
        "Returns the model of node v"
        if v is None:
            v = self.estopping
        hist = self.hist
        trace = self.trace(v)
        m = Model(trace, hist, nvar=self._base._nvar,
                  classifier=self._classifier, labels=self._labels)
        return m

    def trace(self, n):
        "Restore the position in the history of individual v's nodes"
        trace_map = {}
        self._trace(n, trace_map)
        s = list(trace_map.keys())
        s.sort()
        return s

    def _trace(self, n, trace_map):
        if n.position in trace_map:
            return
        else:
            trace_map[n.position] = 1
        if isinstance(n, Function) and n.height > 0:
            if isinstance(n.variable, list):
                for x in n.variable:
                    self._trace(self.hist[x], trace_map)
            else:
                self._trace(self.hist[n.variable], trace_map)

    def clean(self, v):
        self._density -= self.get_density(v)
        if self.estopping is not None and v == self.estopping:
            return
        v.y = None
        v.hy = None

    def random_selection(self, negative=False):
        return np.random.randint(self.popsize)

    def random(self):
        res = []
        done = {}
        randint = np.random.randint
        popsize = self.popsize
        for _ in range(self._tournament_size):
            k = randint(popsize)
            while k in done:
                k = randint(popsize)
            done[k] = 1
            res.append(k)
        return res

    def tournament(self, negative=False):
        """Tournament selection and when negative is True it performs negative
        tournament selection"""
        if self.generation <= self._random_generations:
            return self.random_selection(negative=negative)
        vars = self.random()
        fit = [(k, self.population[x].fitness) for k, x in enumerate(vars)]
        if negative:
            fit = min(fit, key=lambda x: x[1])
        else:
            fit = max(fit, key=lambda x: x[1])
        index = fit[0]
        return vars[index]

    def variable_input(self, used_inputs):
        base = self._base
        for _ in range(base._number_tries_feasible_ind):
            if used_inputs.empty():
                return None
            var = base._random_leaf(used_inputs.get_one())
            if var is not None:
                return var
        return None

    def create_population(self):
        "Create the initial population"
        base = self._base
        if base._share_inputs:
            used_inputs_var = SelectNumbers([x for x in range(base.nvar)])
            used_inputs_naive = used_inputs_var
        if base._pr_variable == 0:
            used_inputs_var = SelectNumbers([])
            used_inputs_naive = SelectNumbers([x for x in range(base.nvar)])
        elif base._pr_variable == 1:
            used_inputs_var = SelectNumbers([x for x in range(base.nvar)])
            used_inputs_naive = SelectNumbers([])
        else:
            used_inputs_var = SelectNumbers([x for x in range(base.nvar)])
            used_inputs_naive = SelectNumbers([x for x in range(base.nvar)])
        nb_input = Inputs(base, used_inputs_naive, functions=base._input_functions)
        while (base._all_inputs or
               (self.popsize < base.popsize and
                not base.stopping_criteria())):
            if base._all_inputs and used_inputs_var.empty() and used_inputs_naive.empty():
                base._init_popsize = self.popsize
                break
            if not used_inputs_var.empty() and np.random.random() < base._pr_variable:
                v = self.variable_input(used_inputs_var)
                if v is None:
                    used_inputs_var.pos = used_inputs_var.size
                    continue
            elif not used_inputs_naive.empty():
                v = nb_input.input()
                # v = self.naive_bayes_input(density, unique_individuals, used_inputs_naive)
                if not used_inputs_var.empty() and used_inputs_naive.empty():
                    base._pr_variable = 1
                if v is None:
                    used_inputs_naive.pos = used_inputs_naive.size
                    if not used_inputs_var.empty():
                        base._pr_variable = 1
                    continue
            else:
                gen = self.generation
                self.generation = 0
                v = base.random_offspring()
                self.generation = gen
            self.add(v)

    def add(self, v):
        "Add an individual to the population"
        self.population.append(v)
        self._current_popsize += 1
        v.position = len(self._hist)
        self._hist.append(v)
        self.bsf = v
        self.estopping = v
        self._density += self.get_density(v)

    def replace(self, v):
        """Replace an individual selected by negative tournament selection with
        individual v"""
        if self.popsize < self._popsize:
            return self.add(v)
        k = self.tournament(negative=True)
        self.clean(self.population[k])
        self.population[k] = v
        v.position = len(self._hist)
        self._hist.append(v)
        self.bsf = v
        self.estopping = v
        self._inds_replace += 1
        self._density += self.get_density(v)
        if self._inds_replace == self._popsize:
            self._inds_replace = 0
            self.generation += 1
            gc.collect()


class SteadyState(BasePopulation):
    def create_population(self):
        super(SteadyState, self).create_population()
        if self.popsize > self._popsize:
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            [self.clean(x) for x in self.population[self._popsize:]]
            self.population = self.population[:self._popsize]


class Generational(BasePopulation):
    "Generational GP using a steady-state as base"
    def __init__(self, *args, **kwargs):
        self._inner = []
        super(Generational, self).__init__(*args, **kwargs)

    def replace(self, v):
        if self.popsize < self._popsize:
            return self.add(v)
        v.position = len(self._hist)
        self._hist.append(v)
        self.bsf = v
        self.estopping = v
        self._inner.append(v)
        if len(self._inner) == self._popsize:
            [self.clean(x) for x in self.population]
            self.population = self._inner
            self._inner = []
            self.generation += 1
            gc.collect()


class HGenerational(Generational):
    def input(self, vars):
        base = self._base
        function_set = base.function_set
        function_selection = base._function_selection_ins
        function_selection.density = base.population.density
        function_selection.unfeasible_functions.clear()
        for i in range(base._number_tries_feasible_ind):
            if base._function_selection:
                func_index = function_selection.tournament()
            else:
                func_index = function_selection.random_function()
            func = function_set[func_index]
            args = vars.get(func.nargs)
            try:
                if len(args) < func.min_nargs:
                    return None
            except AttributeError:
                if len(args) < func.nargs:
                    return None
            args = [self.population[x].position for x in args]
            f = base._random_offspring(func, args)
            if f is None:
                vars.pos -= len(args)
                function_selection.unfeasible_functions.add(func_index)
                continue
            function_selection[func_index] = f.fitness
            return f
        return None

    def extra_inputs(self):
        base = self._base
        if self.popsize <= self._popsize:
            base._init_popsize = self.popsize
            return
        previous = 0
        end = self.popsize
        nvar = end
        while nvar > self._popsize:
            _ = [x for x in range(previous, end)]
            vars = SelectNumbers(_)
            while True:
                f = self.input(vars)
                if f is not None:
                    self.add(f)
                else:
                    break
            nvar = self.popsize - end
            previous = end
            end = self.popsize
        base._unfeasible_counter = 0
        base._init_popsize = self.popsize
        if self.popsize > self._popsize:
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            [self.clean(x) for x in self.population[self._popsize:]]
            self.population = self.population[:self._popsize]

    def create_population(self):
        base = self._base
        random = np.random.random
        pr_variable = base._pr_variable
        assert pr_variable < 1
        nvar = base.nvar
        _ = [x for x in range(nvar)]
        used_inputs_naive = SelectNumbers(_)
        nb_input = Inputs(base, used_inputs_naive,
                          functions=base._input_functions)
        variable_input = self.variable_input
        add = self.add
        input = nb_input.input
        while not used_inputs_naive.empty():
            if pr_variable > 0 and random() < pr_variable:
                v = variable_input(used_inputs_naive)
            else:
                v = input()
            if v is None:
                used_inputs_naive.pos = used_inputs_naive.size
                continue
            add(v)
        return self.extra_inputs()
