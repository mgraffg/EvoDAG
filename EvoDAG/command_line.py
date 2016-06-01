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
from EvoDAG.utils import RandomParameterSearch, PARAMS
from EvoDAG.model import Ensemble
from multiprocessing import Pool
from EvoDAG import EvoDAG
import os
import gzip
import pickle
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


def init_evodag(seed_args_X_y_test):
    seed, args, X, y, test = seed_args_X_y_test
    m = EvoDAG(seed=seed, **args).fit(X, y, test_set=test)
    return m.model()


def rs_evodag(args_X_y):
    args, X, y = args_X_y
    rs = RandomParameterSearch
    fit = []
    for seed in range(3):
        evo = EvoDAG(seed=seed,
                     **rs.process_params(args)).fit(X, y)
        fit.append(evo.model().fitness_vs)
    return fit, args


class CommandLine(object):
    def __init__(self):
        self.Xtest = None
        self.word2id = {}
        self.parser = argparse.ArgumentParser(description="EvoDAG")
        self.training_set()
        self.init_params()
        self.model_file()
        self.test_set()
        self.optimize_parameters()
        self.cores()
        self.ensemble()
        self.output_file()

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
                                 default=1,
                                 type=int)

    def cores(self):
        self.parser.add_argument('-u', '--cpu-cores',
                                 help='Number of cores',
                                 dest='cpu_cores',
                                 default=1,
                                 type=int)

    def optimize_parameters(self):
        cdn = '''Optimize parameters sampling
        N points from the parameter space'''
        self.parser.add_argument('-r', '--optimize-parameters',
                                 dest='optimize_parameters',
                                 type=int, help=cdn)
        cdn = 'File to store the fitness of the configuration explored'
        self.parser.add_argument('--cache-file',
                                 dest='cache_file',
                                 type=str,
                                 help=cdn)

    def test_set(self):
        cdn = 'File containing the test set on csv.'
        self.parser.add_argument('-t', '--test_set',
                                 default=None, type=str,
                                 help=cdn)

    def model_file(self):
        pa = self.parser.add_argument
        pa('-m', '--model-file', type=str, dest='model_file',
           default=None,
           help='File name to either store or load the model')

    def init_params(self):
        pa = self.parser.add_argument
        pa('-c', '--classifier', dest='classifier',
           help='The task is either classification or regression',
           default=True, type=bool)
        pa('-e', '--early_stopping_rounds', dest='early_stopping_rounds',
           type=int,
           help='Early stopping rounds')
        pa('-p', '--popsize', dest='popsize',
           type=int, help='Population size')
        pa('-s', '--seed', dest='seed',
           default=0,
           type=int, help='Seed')

    def training_set(self):
        cdn = 'File containing the training set on csv.'
        self.parser.add_argument('training_set',
                                 nargs='?',
                                 default=None,
                                 help=cdn)

    def parse_args(self):
        self.data = self.parser.parse_args()
        self.main()

    def convert(self, x):
        try:
            return float(x)
        except ValueError:
            if x not in self.word2id:
                self.word2id[x] = len(self.word2id)
            return self.word2id[x]

    def read_data(self, fname):
        with open(fname, 'r') as fpt:
            l = fpt.readlines()
        X = []
        for i in l:
            x = i.rstrip().lstrip()
            if len(x):
                X.append([self.convert(i) for i in x.split(',')])
        return X

    def read_training_set(self):
        if self.data.training_set is None:
            return
        d = np.array(self.read_data(self.data.training_set))
        self.X = d[:, :-1]
        self.y = d[:, -1]

    def read_test_set(self):
        if self.data.test_set is None:
            return False
        d = np.array(self.read_data(self.data.test_set))
        self.Xtest = d
        return True

    def get_model_file(self):
        if self.data.model_file is None:
            a = self.data.training_set.split('.')[0]
            self.data.model_file = a + '.evodag.gz'
        return self.data.model_file

    def store_model(self, kw):
        if self.data.ensemble_size == 1:
            self.evo = EvoDAG(**kw).fit(self.X, self.y)
            self.model = self.evo.model()
        else:
            seed = self.data.seed
            esize = self.data.ensemble_size
            args = [(x, kw, self.X, self.y, self.Xtest)
                    for x in range(seed, seed+esize)]
            if self.data.cpu_cores == 1:
                evo = [init_evodag(x) for x in tqdm(args, total=len(args))]
            else:
                p = Pool(self.data.cpu_cores)
                evo = [x for x in tqdm(p.imap_unordered(init_evodag, args),
                                       total=len(args))]
                p.close()
            self.model = Ensemble(evo)
        model_file = self.get_model_file()
        with gzip.open(model_file, 'w') as fpt:
            pickle.dump(self.model, fpt)
            pickle.dump(self.word2id, fpt)

    def evolve(self, kw):
        if os.path.isfile(self.get_model_file()):
            with gzip.open(self.get_model_file(), 'r') as fpt:
                self.model = pickle.load(fpt)
                self.word2id = pickle.load(fpt)
                return
        if self.data.optimize_parameters is not None:
            if len(kw):
                params = PARAMS.copy()
                for k, v in kw.items():
                    if k in params:
                        params[k] = [v]
            cache_file = self.data.cache_file
            if cache_file is not None and os.path.isfile(cache_file):
                with gzip.open(self.data.cache_file, 'r') as fpt:
                    res = pickle.load(fpt)
            else:
                npoints = self.data.optimize_parameters
                rs = RandomParameterSearch(params=params,
                                           seed=self.data.seed,
                                           npoints=npoints)
                if self.data.cpu_cores == 1:
                    res = [rs_evodag((args, self.X, self.y))
                           for args in tqdm(rs, total=rs._npoints)]
                else:
                    p = Pool(self.data.cpu_cores)
                    args = [(args, self.X, self.y) for args in rs]
                    res = [x for x in tqdm(p.imap_unordered(rs_evodag, args),
                                           total=len(args))]
                    p.close()
                res.sort(key=lambda x: np.median(x[0]), reverse=True)
                if self.data.cache_file is not None:
                    with gzip.open(self.data.cache_file, 'w') as fpt:
                        pickle.dump(res, fpt)
            kw = RandomParameterSearch.process_params(res[0][1])
        self.store_model(kw)

    def get_output_file(self):
        if self.data.output_file is None:
            self.data.output_file = self.data.test_set + '.evodag.csv'
        return self.data.output_file

    def id2word(self, x):
        if not self.data.classifier:
            return x
        if len(self.word2id) == 0:
            return x
        i2w = dict([(i[1], i[0]) for i in self.word2id.items()])
        return [i2w[int(i)] for i in x]

    def main(self):
        self.read_training_set()
        test_set = self.read_test_set()
        kw = {}
        for k, v in EvoDAG().get_params().items():
            if hasattr(self.data, k):
                kw[k] = getattr(self.data, k)
        self.evolve(kw)
        if test_set:
            hy = self.id2word(self.model.predict(self.Xtest))
            with open(self.get_output_file(), 'w') as fpt:
                fpt.write('\n'.join(map(str, hy)))


def main():
    "Command line main"
    c = CommandLine()
    c.parse_args()
