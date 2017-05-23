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
import argparse
import numpy as np
from .utils import RandomParameterSearch, PARAMS
from SparseArray import SparseArray
from .model import Ensemble
import collections
import os
from multiprocessing import Pool
import EvoDAG as evodag
from EvoDAG import EvoDAG
from .utils import tonparray
import time
import gzip
import json
import logging
import gc
import pickle
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


def init_evodag(seed_args_X_y_test):
    seed, args, X, y, test = seed_args_X_y_test
    m = EvoDAG(seed=seed, **args).fit(X, y, test_set=test)
    m = m.model()
    gc.collect()
    return m


def rs_evodag(args_X_y):
    args, X, y = args_X_y
    rs = RandomParameterSearch
    fit = []
    init = time.time()
    for seed in range(3):
        try:
            evo = EvoDAG(seed=seed,
                         **rs.process_params(args)).fit(X, y)
            fit.append(evo.model().fitness_vs)
        except RuntimeError:
            fit.append(-np.inf)
    args['_time'] = time.time() - init
    gc.collect()
    return fit, args


def get_model_fitness(fname):
    with gzip.open(fname) as fpt:
        _ = pickle.load(fpt)
        return (fname, _.fitness_vs)


class CommandLine(object):
    def version(self):
        pa = self.parser.add_argument
        pa('--version',
           action='version', version='EvoDAG %s' % evodag.__version__)
        pa('--verbose', dest='verbose', default=logging.NOTSET, type=int)

    def output_file(self):
        self.parser.add_argument('-o', '--output-file',
                                 help='File to store the test set',
                                 dest='output_file',
                                 default=None,
                                 type=str)

    def ensemble(self):
        self.parser.add_argument('-n', '--ensemble-size',
                                 help='Ensemble size',
                                 dest='ensemble_size',
                                 default=30,
                                 type=int)

    def cores(self):
        self.parser.add_argument('-u', '--cpu-cores',
                                 help='Number of cores',
                                 dest='cpu_cores',
                                 default=1,
                                 type=int)

    def test_set(self):
        cdn = 'File containing the test set on csv.'
        self.parser.add_argument('-t', '--test_set',
                                 default=None, type=str,
                                 help=cdn)

    def init_params(self):
        pa = self.parser.add_argument
        g = self.parser.add_mutually_exclusive_group(required=True)
        g.add_argument('-C', '--classifier', dest='classifier',
                       help='The task is classification (default)',
                       default=True,
                       action="store_true")
        g.add_argument('-R', '--regressor', dest='regressor',
                       help='The task is regression',
                       action="store_true")
        pa('-e', '--early_stopping_rounds', dest='early_stopping_rounds',
           type=int,
           help='Early stopping rounds')
        pa('-p', '--popsize', dest='popsize',
           type=int, help='Population size')
        pa('-s', '--seed', dest='seed',
           default=0,
           type=int, help='Seed')
        pa('-j', '--json', dest='json',
           action="store_true",
           help='Whether the inputs are in json format',
           default=False)
        # pa('--evolution', dest='population_class',
        #    help="Type of evolution (SteadyState|Generational)",
        #    type=str)
        pa('--time-limit', dest='time_limit',
           help='Time limit in seconds', type=int)
        # pa('--random-generations', dest='random_generations',
        #    help='Number of random generations', type=int)

    def training_set(self):
        cdn = 'File containing the training set on csv.'
        self.parser.add_argument('training_set',
                                 nargs='?',
                                 default=None,
                                 help=cdn)

    def parse_args(self):
        self.data = self.parser.parse_args()
        if hasattr(self.data, 'regressor') and self.data.regressor:
            self.data.classifier = False
        if hasattr(self.data, 'verbose'):
            logging.basicConfig()
            logger = logging.getLogger('EvoDAG')
            logger.setLevel(self.data.verbose)
            logger.info('Logging to: %s', self.data.verbose)
        self.main()

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

    def read_data(self, fname):
        with open(fname, 'r') as fpt:
            l = fpt.readlines()
        X = []
        for i in l:
            x = i.rstrip().lstrip()
            if len(x):
                X.append([i for i in x.split(',')])
        return X

    @staticmethod
    def _num_terms(a):
        if 'num_terms' in a:
            num_terms = a['num_terms']
        else:
            num_terms = len(a)
            if 'klass' in a:
                num_terms -= 1
        return num_terms

    def read_data_json_vec(self, fname):
        import json
        X = None
        y = []
        dependent = os.getenv('KLASS')
        if dependent is None:
            dependent = 'klass'
        if fname.endswith('.gz'):
            with gzip.open(fname, 'rb') as fpt:
                l = fpt.readlines()
        else:
            with open(fname, 'r') as fpt:
                l = fpt.readlines()
        for row, d in enumerate(l):
            try:
                a = json.loads(str(d, encoding='utf-8'))
            except TypeError:
                a = json.loads(d)
            vec = a['vec']
            vecsize = a['vecsize']
            if X is None:
                X = [list() for i in range(vecsize)]
            for k, v in vec:
                k = int(k)
                X[k].append((row, self.convert(v)))
            y.append(self.convert_label(a[dependent]))
        num_rows = len(l)
        X = [SparseArray.index_data(x, num_rows) for x in X]
        if len(y) == 0:
            y = None
        else:
            y = np.array(y)
        return X, y

    def read_data_json(self, fname):
        import json
        X = None
        y = []
        dependent = os.getenv('KLASS')
        if dependent is None:
            dependent = 'klass'
        if fname.endswith('.gz'):
            with gzip.open(fname, 'rb') as fpt:
                l = fpt.readlines()
        else:
            with open(fname, 'r') as fpt:
                l = fpt.readlines()
        flag = True
        for row, d in enumerate(l):
            try:
                a = json.loads(str(d, encoding='utf-8'))
            except TypeError:
                a = json.loads(d)
            if flag and 'vec' in a:
                return self.read_data_json_vec(fname)
            else:
                flag = False
            if X is None:
                X = [list() for i in range(self._num_terms(a))]
            for k, v in a.items():
                try:
                    k = int(k)
                    X[k].append((row, self.convert(v)))
                except ValueError:
                    if k == dependent:
                        y.append(self.convert_label(v))
        num_rows = len(l)
        X = [SparseArray.index_data(x, num_rows) for x in X]
        if len(y) == 0:
            y = None
        else:
            y = np.array(y)
        return X, y

    def read_training_set(self):
        if self.data.training_set is None:
            return
        if not self.data.json:
            d = self.read_data(self.data.training_set)
            X = []
            y = []
            if self.data.output_dim > 1:
                dim = self.data.output_dim
                for x in d:
                    X.append([self.convert(i) for i in x[:-dim]])
                    y.append(x[-dim:])
                self.X = np.array(X)
                self.y = [SparseArray.fromlist([float(x[i]) for x in y]) for i in range(dim)]
            else:
                for x in d:
                    X.append([self.convert(i) for i in x[:-1]])
                    y.append(self.convert_label(x[-1]))
                self.X = np.array(X)
                self.y = np.array(y)
            return True
        else:
            X, y = self.read_data_json(self.data.training_set)
            self.X = X
            self.y = y
            return True

    def read_test_set(self):
        if self.data.test_set is None:
            return False
        if not self.data.json:
            X = self.read_data(self.data.test_set)
            self.Xtest = np.array([[self.convert(i) for i in x] for x in X])
            return True
        else:
            X, _ = self.read_data_json(self.data.test_set)
            self.Xtest = X
            return True

    def get_model_file(self):
        if self.data.model_file is None:
            a = self.data.training_set.split('.')[0]
            self.data.model_file = a + '.evodag.gz'
        return self.data.model_file

    def store_model(self, kw):
        if self.data.ensemble_size == 1:
            if self.data.seed >= 0:
                kw['seed'] = self.data.seed
            self.evo = EvoDAG(**kw).fit(self.X, self.y, test_set=self.Xtest)
            self.model = self.evo.model()
        else:
            min_size = self.data.min_size
            esize = self.data.ensemble_size
            init = self.data.seed
            end = init + esize
            evo = []
            while len(evo) < esize:
                args = [(x, kw, self.X, self.y, self.Xtest)
                        for x in range(init, end)]
                if self.data.cpu_cores == 1:
                    _ = [init_evodag(x) for x in tqdm(args, total=len(args))]
                else:
                    p = Pool(self.data.cpu_cores, maxtasksperchild=1)
                    _ = [x for x in tqdm(p.imap_unordered(init_evodag, args),
                                         total=len(args))]
                    p.close()
                [evo.append(x) for x in _ if x.size >= min_size]
                init = end
                end = init + (esize - len(evo))
            self.model = Ensemble(evo)
        model_file = self.get_model_file()
        with gzip.open(model_file, 'w') as fpt:
            pickle.dump(self.model, fpt)
            pickle.dump(self.word2id, fpt)
            pickle.dump(self.label2id, fpt)

    def get_output_file(self):
        if self.data.output_file is None:
            self.data.output_file = self.data.test_set + '.evodag.csv'
            # if self.data.json:
            #     self.data.output_file += '.json'
            # else:
            #     self.data.output_file += '.csv'
        return self.data.output_file

    def id2label(self, x):
        if not self.data.classifier:
            return x
        if len(self.label2id) == 0:
            return x
        i2w = dict([(i[1], i[0]) for i in self.label2id.items()])
        return [i2w[int(i)] for i in x]

    def main(self):
        pass


class CommandLineParams(CommandLine):
    def __init__(self):
        self.Xtest = None
        self.word2id = {}
        self.label2id = {}
        self.parser = argparse.ArgumentParser(description="EvoDAG")
        self.training_set()
        self.init_params()
        self.optimize_parameters()
        self.cores()
        self.version()

    def optimize_parameters(self):
        cdn = '''Optimize parameters sampling
        N (734 by default) points from the parameter space'''
        self.parser.add_argument('-r', '--optimize-parameters',
                                 dest='optimize_parameters',
                                 default=734,
                                 type=int, help=cdn)
        cdn = 'File to store the fitness of the parameters explored'
        self.parser.add_argument('-P', '--parameters',
                                 dest='parameters',
                                 type=str,
                                 help=cdn)
        cdn = 'File containing the parameters values (json) to be explored'
        self.parser.add_argument('--parameters-values',
                                 dest='parameters_values',
                                 type=str,
                                 help=cdn)
        self.parser.add_argument('--output-dim',
                                 dest='output_dim',
                                 default=1,
                                 type=int,
                                 help="Output Dimension (default 1) use with multiple-outputs flag")
        self.parser.add_argument('--only-paramsfiles',
                                 help='Save the params to disk creating a directory',
                                 dest='do_nothing',
                                 action="store_true",
                                 default=False)

    def fs_type_constraint(self, params):
        fs_class = {}
        for x in EvoDAG()._function_set:
            fs_class[x.__name__] = x
        p_delete = []
        for x in params.keys():
            if x in fs_class:
                try:
                    if self.data.classifier:
                        flag = fs_class[x].classification
                    else:
                        flag = fs_class[x].regression
                    if not flag:
                        p_delete.append(x)
                except AttributeError:
                    pass
        for x in p_delete:
            del params[x]

    def if_type_contraint(self, params):
        import importlib
        unique = {}
        if 'input_functions' not in params:
            return
        input_functions = params['input_functions']
        R = []
        for inner in input_functions:
            r = []
            for f in inner:
                _ = importlib.import_module('EvoDAG.node')
                j = getattr(_, f)
                if self.data.classifier:
                    flag = j.classification
                else:
                    flag = j.regression
                if flag:
                    r.append(f)
            if len(r):
                key = ';'.join(r)
                if key not in unique:
                    R.append(r)
                    unique[key] = 1
        if len(R) == 1:
            R = R[0]
        params['input_functions'] = R

    def evolve(self, kw):
        if self.data.parameters_values:
            with open(self.data.parameters_values, 'r') as fpt:
                params = json.loads(fpt.read())
        else:
            params = PARAMS.copy()
        if len(kw):
            for k, v in kw.items():
                if k in params and v is not None:
                    params[k] = [v]
        self.fs_type_constraint(params)
        self.if_type_contraint(params)
        parameters = self.data.parameters
        if parameters is None:
            parameters = self.data.training_set + '.EvoDAGparams'
        npoints = self.data.optimize_parameters
        if isinstance(self.X, list):
            training_size = self.X[0].size()
        else:
            training_size = self.X.shape[0]
        rs = RandomParameterSearch(params=params,
                                   seed=self.data.seed,
                                   training_size=training_size,
                                   npoints=npoints)
        if self.data.do_nothing:
            os.mkdir(parameters)
            for k, x in enumerate(rs):
                fname = os.path.join(parameters, '%s_params.json' % k)
                with open(fname, 'w') as fpt:
                    fpt.write(json.dumps(x, sort_keys=True, indent=2))
            return
        if self.data.cpu_cores == 1:
            res = [rs_evodag((args, self.X, self.y))
                   for args in tqdm(rs, total=rs._npoints)]
        else:
            p = Pool(self.data.cpu_cores, maxtasksperchild=1)
            args = [(args, self.X, self.y) for args in rs]
            res = [x for x in tqdm(p.imap_unordered(rs_evodag, args),
                                   total=len(args))]
            p.close()
        [x[1].update(dict(fitness=x[0])) for x in res]
        res = [x[1] for x in res]
        [x.update(kw) for x in res]
        res.sort(key=lambda x: np.median(x['fitness']), reverse=True)
        res = json.dumps(res, sort_keys=True, indent=2)
        if parameters.endswith('.gz'):
            with gzip.open(parameters, 'wb') as fpt:
                try:
                    fpt.write(bytes(res, encoding='utf-8'))
                except TypeError:
                    fpt.write(res)
        else:
            with open(parameters, 'w') as fpt:
                fpt.write(res)

    def main(self):
        self.read_training_set()
        kw = {}
        if self.data.classifier:
            self.data.multiple_outputs = True
        elif self.data.output_dim > 1:
            self.data.multiple_outputs = True
        for k, v in EvoDAG().get_params().items():
            if hasattr(self.data, k) and getattr(self.data, k) is not None:
                kw[k] = getattr(self.data, k)
        self.evolve(kw)


class CommandLineTrain(CommandLine):
    def __init__(self):
        self.Xtest = None
        self.word2id = {}
        self.label2id = {}
        self.parser = argparse.ArgumentParser(description="EvoDAG")
        self.training_set()
        self.parameters()
        self.model()
        self.cores()
        self.ensemble()
        self.test_set()
        self.version()

    def parameters(self):
        cdn = 'File containing a list of parameters explored,\
        the first one being the best'
        self.parser.add_argument('-P', '--parameters',
                                 dest='parameters',
                                 type=str,
                                 help=cdn)
        self.parser.add_argument('--output-dim',
                                 dest='output_dim',
                                 default=1,
                                 type=int,
                                 help="Output Dimension (default 1) use with multiple-outputs flag")

    def model(self):
        cdn = 'File to store EvoDAG model'
        pa = self.parser.add_argument
        pa('-m', '--model',
           dest='model_file',
           type=str,
           help=cdn)
        pa('-j', '--json', dest='json',
           action="store_true",
           help='Whether the inputs are in json format',
           default=False)
        pa('--min-size', dest='min_size',
           type=int, default=1, help='Model min-size')
        pa('-s', '--seed', dest='seed',
           default=-1, type=int, help='Seed')

    def main(self):
        self.read_training_set()
        self.read_test_set()
        parameters = self.data.parameters
        if parameters.endswith('.gz'):
            func = gzip.open
        else:
            func = open
        with func(parameters, 'rb') as fpt:
            try:
                d = fpt.read()
                res = json.loads(str(d, encoding='utf-8'))
            except TypeError:
                res = json.loads(d)
        try:
            kw = res[0]
        except KeyError:
            kw = res
        kw = RandomParameterSearch.process_params(kw)
        if 'seed' in kw:
            if self.data.seed < 0:
                self.data.seed = kw['seed']
            del kw['seed']
        self.store_model(kw)


class CommandLinePredict(CommandLine):
    def __init__(self):
        self.Xtest = None
        self.word2id = {}
        self.label2id = {}
        self.parser = argparse.ArgumentParser(description="EvoDAG")
        self.model()
        self.test_set()
        self.output_file()
        self.raw_outputs()
        self.cores()
        self.version()

    def test_set(self):
        cdn = 'File containing the test set on csv.'
        self.parser.add_argument('test_set',
                                 default=None,
                                 help=cdn)

    def raw_outputs(self):
        cdn = 'Raw decision function.'
        self.parser.add_argument('--raw-outputs',
                                 default=False,
                                 dest='raw_outputs',
                                 action='store_true',
                                 help=cdn)

    def model(self):
        cdn = 'EvoDAG model'
        pa = self.parser.add_argument
        pa('-m', '--model',
           dest='model_file',
           type=str,
           help=cdn)
        pa('-j', '--json', dest='json',
           action="store_true",
           help='Whether the inputs are in json format',
           default=False)
        pa('--decision-function', dest='decision_function', default=False,
           action='store_true',
           help='Outputs the decision functions instead of the class')

    def main(self):
        model_file = self.get_model_file()
        with gzip.open(model_file, 'r') as fpt:
            m = pickle.load(fpt)
            self.word2id = pickle.load(fpt)
            self.label2id = pickle.load(fpt)
        self.read_test_set()
        self.data.classifier = m.classifier
        if self.data.raw_outputs:
            hy = m.raw_outputs(self.Xtest,
                               cpu_cores=self.data.cpu_cores)
            if hy.ndim == 3:
                hy.shape = (hy.shape[1] * hy.shape[0], hy.shape[-1])
            hy = "\n".join([",".join([str(i) for i in x]) for x in hy.T])
        elif self.data.decision_function:
            hy = m.decision_function(self.Xtest, cpu_cores=self.data.cpu_cores)
            if isinstance(hy, SparseArray):
                hy = tonparray(hy)
                hy = "\n".join(map(str, hy))
            else:
                hy = np.array([tonparray(x) for x in hy]).T
                hy = "\n".join([",".join([str(i) for i in x]) for x in hy])
        else:
            hy = self.id2label(m.predict(self.Xtest, cpu_cores=self.data.cpu_cores))
            hy = "\n".join(map(str, hy))
        with open(self.get_output_file(), 'w') as fpt:
            fpt.write(hy)


class CommandLineUtils(CommandLine):
    def __init__(self):
        self.Xtest = None
        self.word2id = {}
        self.label2id = {}
        self.parser = argparse.ArgumentParser(description="EvoDAG")
        self.model()
        self.graphviz()
        self.params_stats()
        self.output_file()
        self.fitness()
        self.size()
        self.height()
        self.remove_terminals()
        self.used_inputs_number()
        self.create_ensemble_params()
        self.cores()
        self.version()

    def create_ensemble_params(self):
        self.parser.add_argument('--create-ensemble',
                                 help='Models to ensemble', dest='ensemble',
                                 default=False, action='store_true')
        self.parser.add_argument('--best-params-file',
                                 help='Search for the best configuration in a given directory', dest='best_params_file', default=False, action='store_true')
        self.parser.add_argument('-n', '--ensemble-size',
                                 help='Ensemble size (default: select all models)',
                                 dest='ensemble_size',
                                 default=-1,
                                 type=int)

    def used_inputs_number(self):
        self.parser.add_argument('--used-inputs-number',
                                 help='Number of inputs used',
                                 dest='used_inputs_number',
                                 default=False, action='store_true')

    def remove_terminals(self):
        self.parser.add_argument('--remove-terminals',
                                 help='Do not display terminals',
                                 dest='remove_terminals',
                                 default=False, action='store_true')

    def height(self):
        self.parser.add_argument('--height',
                                 help='Model height',
                                 dest='height',
                                 default=False, action='store_true')

    def size(self):
        self.parser.add_argument('--size',
                                 help='Model size',
                                 dest='size',
                                 default=False, action='store_true')

    def fitness(self):
        self.parser.add_argument('--fitness',
                                 help='Fitness in the validation set',
                                 dest='fitness',
                                 default=False, action='store_true')

    def graphviz(self):
        self.parser.add_argument('-G', '--graphviz',
                                 help='Plot the model using dot language',
                                 dest='graphviz',
                                 default=False, action='store_true')

    def params_stats(self):
        self.parser.add_argument('-P', '--params-stats',
                                 help='Parameters statistics',
                                 dest='params_stats',
                                 default=False, action='store_true')

    def output_file(self):
        self.parser.add_argument('-o', '--output-file',
                                 help='File / directory to store the result(s)',
                                 dest='output_file',
                                 default=None,
                                 type=str)

    def model(self):
        cdn = 'File containing the model/params.'
        self.parser.add_argument('model_file',
                                 default=None,
                                 type=str,
                                 help=cdn)

    def read_params(self, parameters):
        if parameters.endswith('.gz'):
            with gzip.open(parameters, 'rb') as fpt:
                try:
                    res = fpt.read()
                    return json.loads(str(res, encoding='utf-8'))
                except TypeError:
                    return json.loads(res)
        else:
            with open(parameters, 'r') as fpt:
                return json.loads(fpt.read())

    def create_ensemble(self, model_file):
        from glob import glob
        models = []
        for fname in model_file.split(' '):
            for k in tqdm(glob(fname)):
                with gzip.open(k, 'r') as fpt:
                    models.append(pickle.load(fpt))
                    self.word2id = pickle.load(fpt)
                    self.label2id = pickle.load(fpt)
        models.sort(key=lambda x: x.fitness_vs, reverse=True)
        if self.data.ensemble_size > 0:
            models = models[:self.data.ensemble_size]
        self.model = Ensemble(models)
        model_file = self.data.output_file
        with gzip.open(model_file, 'w') as fpt:
            pickle.dump(self.model, fpt)
            pickle.dump(self.word2id, fpt)
            pickle.dump(self.label2id, fpt)

    def get_best_params(self, model_file):
        from glob import glob
        h = {}
        args = glob('%s/*.model' % model_file)
        if self.data.cpu_cores == 1:
            res = [get_model_fitness(x) for x in tqdm(args)]
        else:
            p = Pool(self.data.cpu_cores)
            res = [x for x in tqdm(p.imap_unordered(get_model_fitness, args),
                                   total=len(args))]
            p.close()
        for m, fit in res:
            basename = (m.split(model_file + '/')[1]).split('_')[:1]
            fname = "_".join(basename)
            try:
                h[fname].append(fit)
            except KeyError:
                h[fname] = [fit]
        b = max(h.items(), key=lambda x: np.median(x[1]))
        self.best_params = b[0] + '_params.json'

    def main(self):
        def most_common(K, a):
            try:
                str_type = unicode
            except NameError:
                str_type = str

            l = a.most_common()
            if len(l):
                if len(PARAMS[K]) <= 2:
                    return l[0]
                elif isinstance(l[0][0], str_type):
                    return l[0]
                else:
                    num = np.sum([x * y for x, y in a.items()])
                    den = float(np.sum([y for y in a.values()]))
                    return num / den
            return ""

        model_file = self.get_model_file()
        if self.data.graphviz:
            with gzip.open(model_file, 'r') as fpt:
                m = pickle.load(fpt)
                self.word2id = pickle.load(fpt)
                self.label2id = pickle.load(fpt)
            remove_terminals = self.data.remove_terminals
            if remove_terminals:
                m.graphviz(self.data.output_file, terminals=False)
            else:
                m.graphviz(self.data.output_file)
        elif self.data.params_stats:
            params = {k: collections.Counter() for k in PARAMS.keys()}
            stats = self.read_params(model_file)
            for l in stats:
                for k, v in l.items():
                    if k not in params:
                        continue
                    params[k][v] += 1
            with open(self.data.output_file, 'w') as fpt:
                fpt.write(json.dumps({k: most_common(k, v) for k, v
                                      in params.items()}, sort_keys=True, indent=2))
        elif self.data.fitness:
            with gzip.open(model_file, 'r') as fpt:
                m = pickle.load(fpt)
                self.word2id = pickle.load(fpt)
                self.label2id = pickle.load(fpt)
            print("Median fitness: %0.4f" % (m.fitness_vs * -1))
        elif self.data.size:
            with gzip.open(model_file, 'r') as fpt:
                m = pickle.load(fpt)
                self.word2id = pickle.load(fpt)
                self.label2id = pickle.load(fpt)
            print("Size: %s" % m.size)
        elif self.data.height:
            with gzip.open(model_file, 'r') as fpt:
                m = pickle.load(fpt)
                self.word2id = pickle.load(fpt)
                self.label2id = pickle.load(fpt)
            print("Height: %s" % m.height)
        elif self.data.used_inputs_number:
            with gzip.open(model_file, 'r') as fpt:
                m = pickle.load(fpt)
                self.word2id = pickle.load(fpt)
                self.label2id = pickle.load(fpt)
            inputs = m.inputs()
            print("Used inputs number", len(inputs))
        elif self.data.ensemble:
            self.create_ensemble(model_file)
        elif self.data.best_params_file:
            self.get_best_params(model_file)
            print(self.best_params)


def params(output=False):
    "EvoDAG-params command line"
    c = CommandLineParams()
    c.parse_args()
    if output:
        return c


def train(output=False):
    "EvoDAG-params command line"
    c = CommandLineTrain()
    c.parse_args()
    if output:
        return c


def predict():
    "EvoDAG-params command line"
    c = CommandLinePredict()
    c.parse_args()


def utils(output=False):
    "EvoDAG-utils command line"
    c = CommandLineUtils()
    c.parse_args()
    if output:
        return c
