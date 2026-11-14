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
from .model import Ensemble, EvoDAG as model_EvoDAG, EvoDAGE
import collections
import os
from multiprocessing import Pool
import EvoDAG as evodag
from EvoDAG import EvoDAG
from .utils import tonparray
from .utils import Inputs
import time
import gzip
import json
import logging
import gc
import pickle
DEFAULT_PARAMETERS = os.path.join(os.path.dirname(__file__),
                                  'conf', 'default_parameters.json')
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


def init_evodag(seed_args_X_y_test):
    """Fit and return the model of one seed; the argument is ``(seed, args, X, y, test, dirname)``.

    A model already stored in ``dirname`` is loaded instead of trained.
    """
    seed, args, X, y, test, dirname = seed_args_X_y_test
    if dirname is not None:
        output = os.path.join(dirname, '%s.evodag' % seed)
        if os.path.isfile(output):
            with gzip.open(output) as fpt:
                try:
                    return pickle.load(fpt)
                except Exception:
                    pass
    m = EvoDAG(seed=seed, **args).fit(X, y, test_set=test)
    m = m.model()
    gc.collect()
    if dirname is not None:
        with gzip.open(output, 'w') as fpt:
            pickle.dump(m, fpt)
    return m


def rs_evodag(args_X_y):
    """Evaluate one configuration of the parameter search; the argument is ``(args, X, y)``.

    It trains with seeds 0, 1 and 2 and returns ``(fitness, args)``, the
    three validation fitness values (minus infinity for a failed run) and
    the configuration with the time used in ``_time``.
    """
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
    """Load the model stored in the gzip file ``fname`` and return ``(fname, fitness_vs)``."""
    with gzip.open(fname) as fpt:
        _ = pickle.load(fpt)
        return (fname, _.fitness_vs)


class CommandLine(object):
    """Base class of the four command line tools.

    It adds the arguments that the tools share to an ``argparse`` parser,
    reads the data files and, in ``parse_args``, parses ``sys.argv`` and
    calls ``main``, which each tool defines. The classes are the
    implementation of the console commands, which are the functions
    ``params``, ``train``, ``predict`` and ``utils``.
    """
    def version(self):
        """Add the options ``--version`` and ``--verbose``."""
        pa = self.parser.add_argument
        pa('--version',
           action='version', version='EvoDAG %s' % evodag.__version__)
        pa('--verbose', dest='verbose', default=logging.NOTSET, type=int)

    def output_file(self):
        """Add the option ``-o``/``--output-file``."""
        self.parser.add_argument('-o', '--output-file',
                                 help='File to store the test set',
                                 dest='output_file',
                                 default=None,
                                 type=str)

    def ensemble(self):
        """Add the option ``-n``/``--ensemble-size``."""
        self.parser.add_argument('-n', '--ensemble-size',
                                 help='Ensemble size',
                                 dest='ensemble_size',
                                 default=30,
                                 type=int)

    def cores(self):
        """Add the option ``-u``/``--cpu-cores``."""
        self.parser.add_argument('-u', '--cpu-cores',
                                 help='Number of cores',
                                 dest='cpu_cores',
                                 default=1,
                                 type=int)

    def test_set(self):
        """Add the option ``-t``/``--test_set``."""
        cdn = 'File containing the test set on csv.'
        self.parser.add_argument('-t', '--test_set',
                                 default=None, type=str,
                                 help=cdn)

    def init_params(self):
        """Add the task (``-C``/``-R``) and the options that fix a parameter of the search."""
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
        pa('--time-limit', dest='time_limit',
           help='Time limit in seconds', type=int)

    def training_set(self):
        """Add the positional argument with the training file."""
        cdn = 'File containing the training set on csv.'
        self.parser.add_argument('training_set',
                                 nargs='?',
                                 default=None,
                                 help=cdn)

    def parse_args(self):
        """Parse ``sys.argv``, set the logging level and run ``main``."""
        self.data = self.parser.parse_args()
        if hasattr(self.data, 'regressor') and self.data.regressor:
            self.data.classifier = False
        if hasattr(self.data, 'verbose'):
            logging.basicConfig()
            logger = logging.getLogger('EvoDAG')
            logger.setLevel(self.data.verbose)
            logger.info('Logging to: %s', self.data.verbose)
        self.main()

    def read_training_set(self):
        """Read the training file (CSV or, with ``--json``, JSON) into ``X`` and ``y``."""
        if self.data.training_set is None:
            return
        if not self.data.json:
            self.X, self.y = self.inputs.read_csv(self.data.training_set, self.data.output_dim)
            return True
        else:
            self.X, self.y = self.inputs.read_data_json(self.data.training_set)
            return True

    def read_test_set(self):
        """Read the test file (inputs only) into ``Xtest``; return whether there is one."""
        if self.data.test_set is None:
            return False
        if not self.data.json:
            self.Xtest, _ = self.inputs.read_csv(self.data.test_set, 0)
            return True
        else:
            self.Xtest, _ = self.inputs.read_data_json(self.data.test_set)
            return True

    def get_model_file(self):
        """Name of the model file; by default the training file up to its first dot plus ``.evodag.gz``."""
        if self.data.model_file is None:
            a = self.data.training_set.split('.')[0]
            self.data.model_file = a + '.evodag.gz'
        return self.data.model_file

    def get_output_file(self):
        """Name of the output file; by default the test file plus ``.evodag.csv``."""
        if self.data.output_file is None:
            self.data.output_file = self.data.test_set + '.evodag.csv'
            # if self.data.json:
            #     self.data.output_file += '.json'
            # else:
            #     self.data.output_file += '.csv'
        return self.data.output_file

    def id2label(self, x):
        """Translate predicted classes back to the labels of the training file."""
        if not self.data.classifier:
            return x
        if len(self.label2id) == 0:
            return x
        i2w = dict([(i[1], i[0]) for i in self.label2id.items()])
        return [i2w[int(i)] for i in x]

    def main(self):
        """Run the tool; each command overrides it."""
        pass


class CommandLineParams(CommandLine):
    """Implementation of ``EvoDAG-params``: random search on the parameter space."""
    def __init__(self):
        self.Xtest = None
        self.inputs = Inputs()
        self.word2id = self.inputs.word2id
        self.label2id = self.inputs.label2id
        self.parser = argparse.ArgumentParser(description="EvoDAG")
        self.training_set()
        self.init_params()
        self.optimize_parameters()
        self.cores()
        self.version()

    def optimize_parameters(self):
        """Add the options of the search: ``-r``, ``-P``, ``--parameters-values``, ``--output-dim`` and ``--only-paramsfiles``."""
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
        """Remove from ``params`` the functions that do not apply to the task (classification or regression)."""
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
        """Keep, in ``params['input_functions']``, only the input functions that apply to the task."""
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
        """Sample the parameter space, evaluate each configuration and write them sorted by fitness.

        ``kw`` maps parameters to values that are fixed during the search.
        """
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
            training_size = self.X[0].shape[0]
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
        """Read the training set and run the search."""
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
    """Implementation of ``EvoDAG-train``: train a model or an ensemble and store it."""
    def __init__(self):
        self.Xtest = None
        self.inputs = Inputs()
        self.word2id = self.inputs.word2id
        self.label2id = self.inputs.label2id
        self.parser = argparse.ArgumentParser(description="EvoDAG")
        self.training_set()
        self.parameters()
        self.model()
        self.cores()
        self.ensemble()
        self.test_set()
        self.version()

    def parameters(self):
        """Add the task or parameters-file options (``-C``, ``-R``, ``-P``), ``--output-dim`` and ``--kw``."""
        cdn = 'File containing a list of parameters explored,\
        the first one being the best'
        g = self.parser.add_mutually_exclusive_group(required=True)
        g.add_argument('-C', '--classifier', dest='classifier',
                       help='The task is classification (default)',
                       default=True,
                       action="store_true")
        g.add_argument('-R', '--regressor', dest='regressor',
                       help='The task is regression',
                       action="store_true")
        g.add_argument('-P', '--parameters',
                       dest='parameters',
                       default=None,
                       type=str,
                       help=cdn)
        self.parser.add_argument('--output-dim',
                                 dest='output_dim',
                                 default=1,
                                 type=int,
                                 help="Output Dimension (default 1) use with multiple-outputs flag")
        dr = self.parser.add_argument
        dr('--kw', dest='kwargs', default=None, type=str,
           help='Parameters in json that overwrite default parameters')

    def model(self):
        """Add the options ``-m``, ``-j`` and ``-s``."""
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
        # pa('--min-size', dest='min_size',
        #    type=int, default=1, help='Model min-size')
        pa('-s', '--seed', dest='seed',
           default=-1, type=int, help='Seed')

    def main(self):
        """Train a single model (``-n 1``) or an ensemble and write it, with the word and label maps, to a gzip file."""
        self.read_training_set()
        self.read_test_set()
        model_file = self.get_model_file()
        classifier = False if self.data.regressor else True
        kw = dict(params_fname=self.data.parameters, classifier=classifier)
        if self.data.seed >= 0:
            kw['seed'] = self.data.seed
        if self.data.kwargs is not None:
            _ = json.loads(self.data.kwargs)
            kw.update(_)
        if self.data.ensemble_size == 1:
            evo = model_EvoDAG(**kw).fit(self.X, self.y, test_set=self.Xtest)
            self.model = evo.model
        else:
            self.model = EvoDAGE(n_estimators=self.data.ensemble_size,
                                 n_jobs=self.data.cpu_cores,
                                 tmpdir=model_file + '_dir',
                                 **kw).fit(self.X, self.y,
                                           test_set=self.Xtest)
        with gzip.open(model_file, 'w') as fpt:
            pickle.dump(self.model, fpt)
            pickle.dump(self.word2id, fpt)
            pickle.dump(self.label2id, fpt)


class CommandLinePredict(CommandLine):
    """Implementation of ``EvoDAG-predict``: load a model and predict a file."""
    def __init__(self):
        self.Xtest = None
        self.inputs = Inputs()
        self.word2id = self.inputs.word2id
        self.label2id = self.inputs.label2id
        self.parser = argparse.ArgumentParser(description="EvoDAG")
        self.model()
        self.test_set()
        self.output_file()
        self.raw_outputs()
        self.cores()
        self.version()

    def test_set(self):
        """Add the positional argument with the file to predict."""
        cdn = 'File containing the test set on csv.'
        self.parser.add_argument('test_set',
                                 default=None,
                                 help=cdn)

    def raw_outputs(self):
        """Add the option ``--raw-outputs``."""
        cdn = 'Raw decision function.'
        self.parser.add_argument('--raw-outputs',
                                 default=False,
                                 dest='raw_outputs',
                                 action='store_true',
                                 help=cdn)

    def model(self):
        """Add the options ``-m``, ``-j``, ``--decision-function`` and ``--predict-proba``."""
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
        pa('--predict-proba', dest='predict_proba', default=False,
           action='store_true',
           help='Outputs the probability instead of the class')

    def main(self):
        """Load the model, predict the file and write the classes, values or outputs, one line per example."""
        model_file = self.get_model_file()
        with gzip.open(model_file, 'r') as fpt:
            m = pickle.load(fpt)
            self.inputs.word2id.update(pickle.load(fpt))
            self.inputs.label2id.update(pickle.load(fpt))
        self.read_test_set()
        self.data.classifier = m.classifier
        if self.data.raw_outputs:
            m._n_jobs = self.data.cpu_cores
            hy = m.raw_decision_function(self.Xtest)
            if hy.ndim == 3:
                hy.shape = (hy.shape[1] * hy.shape[0], hy.shape[-1])
            hy = "\n".join([",".join([str(i) for i in x]) for x in hy])
        elif self.data.predict_proba:
            m._n_jobs = self.data.cpu_cores
            hy = m.predict_proba(self.Xtest)
            hy = "\n".join([",".join([str(i) for i in x]) for x in hy])
        elif self.data.decision_function:
            hy = m.decision_function(self.Xtest, cpu_cores=self.data.cpu_cores)
            hy = "\n".join([",".join([str(i) for i in x]) for x in hy])
        else:
            hy = self.id2label(m.predict(self.Xtest, cpu_cores=self.data.cpu_cores))
            hy = "\n".join(map(str, hy))
        with open(self.get_output_file(), 'w') as fpt:
            fpt.write(hy)


class CommandLineUtils(CommandLine):
    """Implementation of ``EvoDAG-utils``: inspect models and parameter files, and combine models."""
    def __init__(self):
        self.Xtest = None
        self.inputs = Inputs()
        self.word2id = self.inputs.word2id
        self.label2id = self.inputs.label2id
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
        """Add the options ``--create-ensemble``, ``--best-params-file`` and ``-n``."""
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
        """Add the option ``--used-inputs-number``."""
        self.parser.add_argument('--used-inputs-number',
                                 help='Number of inputs used',
                                 dest='used_inputs_number',
                                 default=False, action='store_true')

    def remove_terminals(self):
        """Add the option ``--remove-terminals``."""
        self.parser.add_argument('--remove-terminals',
                                 help='Do not display terminals',
                                 dest='remove_terminals',
                                 default=False, action='store_true')

    def height(self):
        """Add the option ``--height``."""
        self.parser.add_argument('--height',
                                 help='Model height',
                                 dest='height',
                                 default=False, action='store_true')

    def size(self):
        """Add the option ``--size``."""
        self.parser.add_argument('--size',
                                 help='Model size',
                                 dest='size',
                                 default=False, action='store_true')

    def fitness(self):
        """Add the option ``--fitness``."""
        self.parser.add_argument('--fitness',
                                 help='Fitness in the validation set',
                                 dest='fitness',
                                 default=False, action='store_true')

    def graphviz(self):
        """Add the option ``-G``/``--graphviz``."""
        self.parser.add_argument('-G', '--graphviz',
                                 help='Plot the model using dot language',
                                 dest='graphviz',
                                 default=False, action='store_true')

    def params_stats(self):
        """Add the option ``-P``/``--params-stats``."""
        self.parser.add_argument('-P', '--params-stats',
                                 help='Parameters statistics',
                                 dest='params_stats',
                                 default=False, action='store_true')

    def output_file(self):
        """Add the option ``-o``/``--output-file``."""
        self.parser.add_argument('-o', '--output-file',
                                 help='File / directory to store the result(s)',
                                 dest='output_file',
                                 default=None,
                                 type=str)

    def model(self):
        """Add the positional argument ``model_file``."""
        cdn = 'File containing the model/params.'
        self.parser.add_argument('model_file',
                                 default=None,
                                 type=str,
                                 help=cdn)

    def read_params(self, parameters):
        """Read a JSON file of parameters (``.gz`` files are read compressed)."""
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
        """Combine the model files (space separated patterns) into one ensemble written to ``-o``."""
        from glob import glob
        models = []
        flag = False
        for fname in model_file.split(' '):
            for k in tqdm(glob(fname)):
                try:
                    with gzip.open(k, 'r') as fpt:
                        models.append(pickle.load(fpt))
                        self.inputs.word2id.update(pickle.load(fpt))
                        self.inputs.label2id.update(pickle.load(fpt))
                except EOFError:
                    flag = True
                    os.unlink(k)
        if flag:
            raise RuntimeError('Unable to read models')
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
        """Find, in a directory of ``K_*.model`` files, the configuration ``K`` with the best median fitness."""
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
            basename = (m.split(os.path.join(model_file, ''))[1]).split('_')[:1]
            fname = "_".join(basename)
            try:
                h[fname].append(fit)
            except KeyError:
                h[fname] = [fit]
        b = max(h.items(), key=lambda x: np.median(x[1]))
        self.best_params = b[0] + '_params.json'

    def main(self):
        """Run the utility selected by the flags."""
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
                self.inputs.word2id.update(pickle.load(fpt))
                self.inputs.label2id.update(pickle.load(fpt))
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
                self.inputs.word2id.update(pickle.load(fpt))
                self.inputs.label2id.update(pickle.load(fpt))
            print("Median fitness: %0.4f" % (m.fitness_vs * -1))
        elif self.data.size:
            with gzip.open(model_file, 'r') as fpt:
                m = pickle.load(fpt)
                self.inputs.word2id.update(pickle.load(fpt))
                self.inputs.label2id.update(pickle.load(fpt))
            print("Size: %s" % m.size)
        elif self.data.height:
            with gzip.open(model_file, 'r') as fpt:
                m = pickle.load(fpt)
                self.inputs.word2id.update(pickle.load(fpt))
                self.inputs.label2id.update(pickle.load(fpt))
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
    """Entry point of the ``EvoDAG-params`` console command.

    It samples configurations from the space of parameters
    (``parameter_values.json``), trains EvoDAG three times with each one
    on the training file and writes the configurations, sorted by the
    median validation fitness, as a JSON list. The arguments are read from
    ``sys.argv``; run ``EvoDAG-params -h`` for them.

    Parameters
    ----------
    output : bool, optional
        If true return the ``CommandLineParams`` instance.

    Notes
    -----
    It reads files and ``sys.argv``, so there is no doctest here. The
    behaviour is verified by ``test_params``, ``test_parameters_values``,
    ``test_optimize_parameters`` and ``test_params_files`` in
    ``tests/test_command_line.py``. See the Command line page of the user
    guide for every option.
    """
    c = CommandLineParams()
    c.parse_args()
    if output:
        return c


def train(output=False):
    """Entry point of the ``EvoDAG-train`` console command.

    It trains a model (``-n 1``) or an ensemble of ``-n`` models, with the
    default parameters or those of a file written by ``EvoDAG-params``, and
    stores it in a gzip file. The arguments are read from ``sys.argv``; run
    ``EvoDAG-train -h`` for them.

    Parameters
    ----------
    output : bool, optional
        If true return the ``CommandLineTrain`` instance.

    Notes
    -----
    It reads files and ``sys.argv``, so there is no doctest here. The
    behaviour is verified by ``test_train``, ``test_train_default_params``,
    ``test_train_kw`` and ``test_train_inner_model`` in
    ``tests/test_command_line.py``.
    """
    c = CommandLineTrain()
    c.parse_args()
    if output:
        return c


def predict():
    """Entry point of the ``EvoDAG-predict`` console command.

    It loads a model written by ``EvoDAG-train`` and writes the predictions
    of a file, or with the options the probabilities, the decision
    function or the outputs of every model, one line per example. The
    arguments are read from ``sys.argv``; run ``EvoDAG-predict -h`` for
    them.

    Notes
    -----
    It reads files and ``sys.argv``, so there is no doctest here. The
    behaviour is verified by ``test_predict``, ``test_predict_proba``,
    ``test_decision_function`` and ``test_raw_outputs_classification`` in
    ``tests/test_command_line.py``.
    """
    c = CommandLinePredict()
    c.parse_args()


def utils(output=False):
    """Entry point of the ``EvoDAG-utils`` console command.

    A set of tools over models and parameter files: draw a model with
    Graphviz, summarise a parameters file, print the fitness, size, height
    or number of inputs used by a model, and combine models into an
    ensemble. The arguments are read from ``sys.argv``; run
    ``EvoDAG-utils -h`` for them.

    Parameters
    ----------
    output : bool, optional
        If true return the ``CommandLineUtils`` instance.

    Notes
    -----
    It reads files and ``sys.argv``, so there is no doctest here. The
    behaviour is verified by ``test_utils_graphviz``,
    ``test_utils_params_stats``, ``test_model_size``, ``test_create_ensemble``
    and ``test_get_best_params_files`` in ``tests/test_command_line.py``.
    """
    c = CommandLineUtils()
    c.parse_args()
    if output:
        return c
