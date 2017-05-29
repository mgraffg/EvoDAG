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

import sys
import os
from test_root import X, cl


def default_nargs():
    from EvoDAG.node import Add, Mul, Div, Min, Max, Atan2, Hypot
    from EvoDAG.node import Fabs, Exp, Sqrt, Sin, Cos, Log1p,\
        Sq, Acos, Asin, Atan,\
        Tan, Cosh, Sinh, Tanh, Acosh, Asinh, Atanh,\
        Expm1, Log, Log2, Log10, Lgamma, Sign, Ceil, Floor,\
        NaiveBayes, NaiveBayesMN
    for f in [Add, Mul, Div, Min, Max, Atan2, Hypot, NaiveBayes, NaiveBayesMN]:
        f.nargs = 2
    for f in [Fabs, Exp, Sqrt, Sin, Cos, Log1p,
              Sq, Acos, Asin, Atan,
              Tan, Cosh, Sinh, Tanh, Acosh, Asinh, Atanh,
              Expm1, Log, Log2, Log10, Lgamma, Sign,
              Ceil, Floor]:
        f.nargs = 1


def training_set():
    import tempfile
    fname = tempfile.mktemp()
    with open(fname, 'w') as fpt:
        for x, v in zip(X, cl):
            l = x.tolist()
            l.append(v)
            fpt.write(','.join(map(str, l)))
            fpt.write('\n')
    return fname


def test_optimize_parameters():
    import os
    from EvoDAG.command_line import params
    fname = training_set()
    sys.argv = ['EvoDAG', '-C', '--parameters',
                'cache.evodag.gz', '-p3', '-e2', '-r2', fname]
    params()
    assert os.path.isfile('cache.evodag.gz')
    os.unlink('cache.evodag.gz')
    os.unlink(fname)
    default_nargs()


def test_cpu_cores():
    import os
    from EvoDAG.command_line import params
    fname = training_set()
    sys.argv = ['EvoDAG', '-C', '-u2', '--parameters',
                'cache.evodag.gz', '-p3', '-e2', '-r', '2', fname]
    params()
    assert os.path.isfile('cache.evodag.gz')
    os.unlink('cache.evodag.gz')
    os.unlink(fname)
    default_nargs()


def test_word2id():
    import tempfile
    from EvoDAG.command_line import params
    fname = tempfile.mktemp()
    id2word = dict([[0, 'a'], [1, 'b'], [2, 'c']])
    with open(fname, 'w') as fpt:
        for x, v in zip(X, cl):
            l = x.tolist()
            l.append(id2word[v])
            fpt.write(','.join(map(str, l)))
            fpt.write('\n')
    sys.argv = ['EvoDAG', '-C', '-Pparams.gz', '-r2', '-e2', '-p3', fname]
    params()
    os.unlink(fname)
    os.unlink('params.gz')
    default_nargs()


def test_json():
    from EvoDAG.command_line import params
    import tempfile
    import json
    fname = tempfile.mktemp()
    with open(fname, 'w') as fpt:
        for x, y in zip(X, cl):
            a = {k: v for k, v in enumerate(x)}
            a['klass'] = int(y)
            a['num_terms'] = len(x)
            fpt.write(json.dumps(a) + '\n')
    print("termine con el json")
    sys.argv = ['EvoDAG', '-C', '-Poutput.evodag', '--json',
                '-e1', '-p3', '-r2', fname]
    params()
    os.unlink(fname)
    print(open('output.evodag').read())
    os.unlink('output.evodag')
    default_nargs()


def test_params():
    import os
    import gzip
    from EvoDAG.command_line import params
    import json
    fname = training_set()
    sys.argv = ['EvoDAG', '-C', '--parameters',
                'cache.evodag.gz', '-p3', '-e2', '-r', '2', fname]
    params()
    os.unlink(fname)
    assert os.path.isfile('cache.evodag.gz')
    with gzip.open('cache.evodag.gz', 'rb') as fpt:
        try:
            d = fpt.read()
            a = json.loads(str(d, encoding='utf-8'))
        except TypeError:
            a = json.loads(d)
    os.unlink('cache.evodag.gz')
    assert len(a) == len([x for x in a if 'fitness' in x])
    print(a)
    default_nargs()


def test_parameters_values():
    import os
    from EvoDAG.command_line import params
    import json
    fname = training_set()
    with open('p.conf', 'w') as fpt:
        fpt.write(json.dumps(dict(popsize=['x'], multiple_outputs=[True])))
    sys.argv = ['EvoDAG', '-C', '-Pcache.evodag.gz', '-p3', '-e2',
                '--parameters-values', 'p.conf',
                '-r', '2', fname]
    try:
        params()
        assert False
    except ValueError:
        pass
    os.unlink('p.conf')
    os.unlink(fname)
    default_nargs()


def test_train():
    import os
    from EvoDAG.command_line import params, CommandLineTrain
    fname = training_set()
    sys.argv = ['EvoDAG', '-C', '-Pcache.evodag.gz',
                '-p3', '-e2', '-r2', fname]
    params()
    sys.argv = ['EvoDAG', '-Pcache.evodag.gz',
                '-n2',
                '--model', 'model.evodag',
                '--test_set', fname, fname]
    c = CommandLineTrain()
    c.parse_args()
    assert os.path.isfile(c.data.test_set)
    assert os.path.isfile('model.evodag')
    os.unlink(fname)
    os.unlink('cache.evodag.gz')
    os.unlink('model.evodag')
    default_nargs()


def test_predict():
    import os
    from EvoDAG.command_line import params, train, predict
    fname = training_set()
    sys.argv = ['EvoDAG', '-C', '--parameters',
                'cache.evodag.gz', '-p3', '-e2', '-r', '2', fname]
    params()
    sys.argv = ['EvoDAG', '--parameters', 'cache.evodag.gz',
                '-n2',
                '--model', 'model.evodag',
                '--test', fname, fname]
    train()
    sys.argv = ['EvoDAG', '--output', 'output.evodag',
                '--model', 'model.evodag', fname]
    predict()
    os.unlink(fname)
    os.unlink('cache.evodag.gz')
    os.unlink('model.evodag')
    assert os.path.isfile('output.evodag')
    os.unlink('output.evodag')
    default_nargs()


def test_generational():
    import os
    from EvoDAG.command_line import CommandLineParams
    import gzip
    import json
    fname = training_set()

    class C(CommandLineParams):
        def parse_args(self):
            self.data = self.parser.parse_args()
            self.data.population_class = 'Generational'
            if hasattr(self.data, 'regressor') and self.data.regressor:
                self.data.classifier = False
            self.main()

    sys.argv = ['EvoDAG', '-C', '--parameters',
                'cache.evodag.gz', '-p3', '-e3',
                '-r', '2', fname]
    c = C()
    c.parse_args()
    with gzip.open('cache.evodag.gz') as fpt:
        data = fpt.read()
        try:
            a = json.loads(str(data, encoding='utf-8'))
        except TypeError:
            a = json.loads(data)
    a = a[0]
    assert 'population_class' in a
    assert a['population_class'] == 'Generational'
    os.unlink('cache.evodag.gz')
    print(a)
    default_nargs()


def test_all_inputs():
    import os
    import json
    from EvoDAG.command_line import CommandLineParams
    fname = training_set()
    sys.argv = ['EvoDAG', '-C', '--parameters',
                'cache.evodag', '-p3', '-e2',
                '-r', '2', fname]
    c = CommandLineParams()
    c.parse_args()
    with open('cache.evodag') as fpt:
        a = json.loads(fpt.read())[0]
    assert 'all_inputs' in a
    os.unlink('cache.evodag')
    default_nargs()


def test_time_limit():
    import os
    from EvoDAG.command_line import params, train
    import json
    fname = training_set()
    sys.argv = ['EvoDAG', '-C', '--parameters',
                'cache.evodag', '-p3', '-e2',
                '--time-limit', '10',
                '-r', '2', fname]
    params()
    sys.argv = ['EvoDAG', '--parameters', 'cache.evodag',
                '-n2',
                '--model', 'model.evodag',
                '--test', fname, fname]
    train()
    os.unlink(fname)
    with open('cache.evodag') as fpt:
        a = json.loads(fpt.read())[0]
    assert 'time_limit' in a
    os.unlink('cache.evodag')
    assert os.path.isfile('model.evodag')
    os.unlink('model.evodag')
    default_nargs()


def test_word2id2():
    import tempfile
    import os
    from EvoDAG.command_line import CommandLineParams
    fname = tempfile.mktemp()
    id2word = dict([[0, 'a'], [1, 'b'], [2, 'c']])
    with open(fname, 'w') as fpt:
        for x, v in zip(X, cl):
            l = x.tolist()
            l.append(id2word[v])
            fpt.write(','.join(map(str, l)))
            fpt.write('\n')
    sys.argv = ['EvoDAG', '-C', '--parameters',
                'cache.evodag', '-p3', '-e1',
                '-r', '2', fname]
    c = CommandLineParams()
    c.parse_args()
    print(len(c.word2id))
    assert len(c.word2id) == 0
    os.unlink('cache.evodag')
    os.unlink(fname)
    default_nargs()


def test_decision_function():
    import os
    from EvoDAG.command_line import params, train, predict
    fname = training_set()
    sys.argv = ['EvoDAG', '-C', '--parameters',
                'cache.evodag', '-p3', '-e1',
                '-r', '1', fname]
    params()
    sys.argv = ['EvoDAG', '--parameters', 'cache.evodag',
                '-n2',
                '--model', 'model.evodag',
                '--test', fname, fname]
    train()
    sys.argv = ['EvoDAG', '--output', 'output.evodag',
                '--decision-function',
                '--model', 'model.evodag', fname]
    predict()
    os.unlink(fname)
    os.unlink('cache.evodag')
    os.unlink('model.evodag')
    os.unlink('output.evodag')
    default_nargs()


def test_random_generations():
    import os
    import json
    from EvoDAG.command_line import params
    fname = training_set()
    sys.argv = ['EvoDAG', '-C', '--parameters',
                'cache.evodag', '-p3', '-e2',
                # '--random-generations', '1',
                '-r', '2', fname]
    params()
    os.unlink(fname)
    with open('cache.evodag') as fpt:
        a = json.loads(fpt.read())[0]
    assert 'random_generations' in a
    os.unlink('cache.evodag')
    default_nargs()


def test_predict_cpu():
    import os
    from EvoDAG.command_line import params, train, predict
    fname = training_set()
    sys.argv = ['EvoDAG', '-C', '--parameters',
                'cache.evodag', '-p3', '-e1',
                '-r2', fname]
    params()
    sys.argv = ['EvoDAG', '--parameters', 'cache.evodag',
                '-n2',
                '--model', 'model.evodag',
                '--test', fname, fname]
    train()
    sys.argv = ['EvoDAG', '--output', 'output.evodag',
                '--decision-function',
                '-u2',
                '--model', 'model.evodag', fname]
    predict()
    os.unlink(fname)
    os.unlink('cache.evodag')
    os.unlink('model.evodag')
    os.unlink('output.evodag')
    default_nargs()


def test_classifier_params():
    import os
    import json
    from EvoDAG.command_line import params
    fname = training_set()
    sys.argv = ['EvoDAG', '-C', '--parameters',
                'cache.evodag', '-p3', '-e1',
                '-r', '2', fname]
    params()
    os.unlink(fname)
    with open('cache.evodag') as fpt:
        a = json.loads(fpt.read())[0]
    assert 'classifier' in a
    assert a['classifier']
    os.unlink('cache.evodag')
    default_nargs()


def test_regressor_params():
    import os
    import json
    from EvoDAG.command_line import params
    fname = training_set()
    sys.argv = ['EvoDAG', '-R', '--parameters',
                'cache.evodag', '-p3', '-e1',
                '-r', '2', fname]
    params()
    os.unlink(fname)
    with open('cache.evodag') as fpt:
        a = json.loads(fpt.read())[0]
    assert 'classifier' in a
    assert not a['classifier']
    assert a['popsize'] == 3
    assert a['early_stopping_rounds'] == 1
    os.unlink('cache.evodag')
    default_nargs()
    

def test_time():
    import os
    import json
    from EvoDAG.command_line import params
    fname = training_set()
    sys.argv = ['EvoDAG', '-R', '--parameters',
                'cache.evodag', '-p3', '-e1',
                '-r', '2', fname]
    params()
    os.unlink(fname)
    with open('cache.evodag') as fpt:
        a = json.loads(fpt.read())[0]
    assert 'classifier' in a
    assert not a['classifier']
    assert a['popsize'] == 3
    assert a['early_stopping_rounds'] == 1
    print(a['_time'])
    assert a['_time'] > 0.001
    os.unlink('cache.evodag')
    default_nargs()


def test_utils_graphviz():
    import os
    from EvoDAG.command_line import params, train, utils
    fname = training_set()
    sys.argv = ['EvoDAG', '-C', '--parameters',
                'cache.evodag', '-p3', '-e1',
                '-r2', fname]
    params()
    sys.argv = ['EvoDAG', '--parameters', 'cache.evodag',
                '-n2',
                '--model', 'model.evodag',
                '--test', fname, fname]
    train()
    sys.argv = ['EvoDAG', '--output', 'output.evodag',
                '--decision-function',
                '-u2',
                '--model', 'model.evodag', fname]
    sys.argv = ['EvoDAG', '-G', '-oevodag_gv',
                'model.evodag']

    utils()
    os.unlink(fname)
    os.unlink('cache.evodag')
    os.unlink('model.evodag')
    for i in range(2):
        os.unlink('evodag_gv/evodag-%s' % i)
    os.rmdir('evodag_gv')
    default_nargs()


def test_json_gzip():
    from EvoDAG.command_line import params
    import gzip
    import tempfile
    import json
    fname = tempfile.mktemp() + '.gz'
    with gzip.open(fname, 'wb') as fpt:
        for x, y in zip(X, cl):
            a = {k: v for k, v in enumerate(x)}
            a['klass'] = int(y)
            a['num_terms'] = len(x)
            try:
                fpt.write(bytes(json.dumps(a) + '\n', encoding='utf-8'))
            except TypeError:
                fpt.write(json.dumps(a) + '\n')
    print("termine con el json")
    sys.argv = ['EvoDAG', '-C', '-Poutput.evodag', '--json',
                '-e1', '-p3', '-r2', fname]
    params()
    os.unlink(fname)
    print(open('output.evodag').read())
    os.unlink('output.evodag')
    default_nargs()


def test_multiple_outputs():
    from EvoDAG.command_line import params
    import gzip
    import tempfile
    import json
    fname = tempfile.mktemp() + '.gz'
    with gzip.open(fname, 'wb') as fpt:
        for x, y in zip(X, cl):
            a = {k: v for k, v in enumerate(x)}
            a['klass'] = int(y)
            a['num_terms'] = len(x)
            try:
                fpt.write(bytes(json.dumps(a) + '\n', encoding='utf-8'))
            except TypeError:
                fpt.write(json.dumps(a) + '\n')
    print("termine con el json")
    sys.argv = ['EvoDAG', '-C', '-Poutput.evodag', '--json',
                '-e1', '-p3', '-r2', fname]
    params()
    os.unlink(fname)
    d = json.loads(open('output.evodag').read())
    assert d[0]['multiple_outputs']
    os.unlink('output.evodag')
    default_nargs()


def mo_training_set():
    import tempfile
    import numpy as np
    klass = np.unique(cl)
    y = np.empty((cl.shape[0], klass.shape[0]))
    y.fill(-1)
    for i, k in enumerate(klass):
        mask = k == cl
        y[mask, i] = 1
    fname = tempfile.mktemp()
    with open(fname, 'w') as fpt:
        for x, v in zip(X, y):
            l = x.tolist()
            l += v.tolist()
            fpt.write(','.join(map(str, l)))
            fpt.write('\n')
    return fname


def test_number_multiple_outputs_classification():
    import os
    from EvoDAG.command_line import params, train, predict
    fname = mo_training_set()
    sys.argv = ['EvoDAG', '--output-dim=3',
                '-C', '--parameters',
                'cache.evodag', '-p3', '-e1',
                '-r2', fname]
    params()
    sys.argv = ['EvoDAG', '--parameters', 'cache.evodag',
                '-n2', '--output-dim=3',
                '--model', 'model.evodag',
                '--test', fname, fname]
    train()
    sys.argv = ['EvoDAG', '--output', 'output.evodag',
                '--decision-function',
                '--model', 'model.evodag', fname]
    predict()
    os.unlink(fname)
    os.unlink('cache.evodag')
    os.unlink('model.evodag')
    l = open('output.evodag').readline()
    os.unlink('output.evodag')
    default_nargs()
    assert len(l.split(',')) == 3


def test_number_multiple_outputs_regression():
    import os
    from EvoDAG.command_line import params, train, predict
    fname = mo_training_set()
    sys.argv = ['EvoDAG', '--output-dim=3',
                '-R', '--parameters',
                'cache.evodag', '-p3', '-e1',
                '-r2', fname]
    params()
    sys.argv = ['EvoDAG', '--parameters', 'cache.evodag',
                '-n2', '--output-dim=3',
                '--model', 'model.evodag',
                '--test', fname, fname]
    train()
    sys.argv = ['EvoDAG', '--output', 'output.evodag',
                '--decision-function',
                '--model', 'model.evodag', fname]
    predict()
    os.unlink(fname)
    os.unlink('cache.evodag')
    os.unlink('model.evodag')
    l = open('output.evodag').readline()
    os.unlink('output.evodag')
    default_nargs()
    assert len(l.split(',')) == 3


def test_utils_params_stats():
    from EvoDAG.command_line import params, utils
    fname = mo_training_set()
    sys.argv = ['EvoDAG', '--output-dim=3',
                '-R', '--parameters',
                'cache.evodag', '-p3', '-e1',
                '-r3', fname]
    params()
    sys.argv = ['EvoDAG', '-P', '-o', fname, 'cache.evodag']
    utils()
    with open(fname) as fpt:
        print(fpt.read())
    os.unlink(fname)
    os.unlink('cache.evodag')
    default_nargs()


def test_model_fitness_vs():
    import os
    from EvoDAG.command_line import params, train, utils
    fname = mo_training_set()
    sys.argv = ['EvoDAG', '--output-dim=3',
                '-R', '--parameters',
                'cache.evodag', '-p3', '-e1',
                '-r2', fname]
    params()
    sys.argv = ['EvoDAG', '--parameters', 'cache.evodag',
                '-n2', '--output-dim=3',
                '--model', 'model.evodag',
                '--test', fname, fname]
    train()
    sys.argv = ['EvoDAG', '--fitness', 'model.evodag']
    utils()
    os.unlink('cache.evodag')
    os.unlink('model.evodag')
    default_nargs()


def test_raw_outputs_regression():
    import os
    from EvoDAG.command_line import params, train, predict
    fname = mo_training_set()
    sys.argv = ['EvoDAG', '--output-dim=3',
                '-R', '--parameters',
                'cache.evodag', '-p3', '-e1',
                '-r2', fname]
    params()
    sys.argv = ['EvoDAG', '--parameters', 'cache.evodag',
                '-n2', '--output-dim=3',
                '--model', 'model.evodag',
                '--test', fname, fname]
    train()
    sys.argv = ['EvoDAG', '--output', 'output.evodag',
                '--raw-outputs',
                '--model', 'model.evodag', fname]
    predict()
    os.unlink(fname)
    os.unlink('cache.evodag')
    os.unlink('model.evodag')
    l = open('output.evodag').readline()
    os.unlink('output.evodag')
    default_nargs()
    print(len(l.split(',')))
    assert len(l.split(',')) == 6


def test_raw_outputs_classification():
    import os
    from EvoDAG.command_line import params, train, predict
    fname = mo_training_set()
    sys.argv = ['EvoDAG', '--output-dim=3',
                '-C', '--parameters',
                'cache.evodag', '-p3', '-e1',
                '-r2', fname]
    params()
    sys.argv = ['EvoDAG', '--parameters', 'cache.evodag',
                '-n2', '--output-dim=3',
                '--model', 'model.evodag',
                '--test', fname, fname]
    train()
    sys.argv = ['EvoDAG', '--output', 'output.evodag',
                '--raw-outputs',
                '--model', 'model.evodag', fname]
    predict()
    os.unlink(fname)
    os.unlink('cache.evodag')
    os.unlink('model.evodag')
    l = open('output.evodag').readline()
    os.unlink('output.evodag')
    default_nargs()
    assert len(l.split(',')) == 6


def test_nan():
    import numpy as np
    from EvoDAG.command_line import CommandLineParams
    c = CommandLineParams()
    assert np.isfinite(c.convert_label('NaN'))
    default_nargs()


def test_model_size():
    import os
    from EvoDAG.command_line import params, train, utils
    fname = mo_training_set()
    sys.argv = ['EvoDAG', '--output-dim=3',
                '-R', '--parameters',
                'cache.evodag', '-p3', '-e1',
                '-r2', fname]
    params()
    sys.argv = ['EvoDAG', '--parameters', 'cache.evodag',
                '-n2', '--output-dim=3',
                '--model', 'model.evodag',
                '--test', fname, fname]
    train()
    sys.argv = ['EvoDAG', '--size', 'model.evodag']
    utils()
    os.unlink('cache.evodag')
    os.unlink('model.evodag')
    default_nargs()


def test_model_height():
    import os
    from EvoDAG.command_line import params, train, utils
    fname = mo_training_set()
    sys.argv = ['EvoDAG', '--output-dim=3',
                '-R', '--parameters',
                'cache.evodag', '-p3', '-e1',
                '-r2', fname]
    params()
    sys.argv = ['EvoDAG', '--parameters', 'cache.evodag',
                '-n2', '--output-dim=3',
                '--model', 'model.evodag',
                '--test', fname, fname]
    train()
    sys.argv = ['EvoDAG', '--height', 'model.evodag']
    utils()
    os.unlink('cache.evodag')
    os.unlink('model.evodag')
    default_nargs()


def test_utils_graphviz_terminals():
    import os
    from EvoDAG.command_line import params, train, utils
    fname = training_set()
    sys.argv = ['EvoDAG', '-C', '--parameters',
                'cache.evodag', '-p3', '-e1',
                '-r2', fname]
    params()
    sys.argv = ['EvoDAG', '--parameters', 'cache.evodag',
                '-n2',
                '--model', 'model.evodag',
                '--test', fname, fname]
    train()
    sys.argv = ['EvoDAG', '--output', 'output.evodag',
                '--decision-function',
                '-u2',
                '--model', 'model.evodag', fname]
    sys.argv = ['EvoDAG', '-G', '-oevodag_gv',
                '--remove-terminals',
                'model.evodag']

    utils()
    os.unlink(fname)
    os.unlink('cache.evodag')
    os.unlink('model.evodag')
    for i in range(2):
        os.unlink('evodag_gv/evodag-%s' % i)
    os.rmdir('evodag_gv')
    default_nargs()


def test_json_gzip_dependent_variable():
    from EvoDAG.command_line import params
    import gzip
    import tempfile
    import json
    import os
    os.environ['KLASS'] = 'dependent'
    fname = tempfile.mktemp() + '.gz'
    with gzip.open(fname, 'wb') as fpt:
        for x, y in zip(X, cl):
            a = {k: v for k, v in enumerate(x)}
            a['dependent'] = int(y)
            a['num_terms'] = len(x)
            try:
                fpt.write(bytes(json.dumps(a) + '\n', encoding='utf-8'))
            except TypeError:
                fpt.write(json.dumps(a) + '\n')
    print("termine con el json")
    sys.argv = ['EvoDAG', '-C', '-Poutput.evodag', '--json',
                '-e1', '-p3', '-r2', fname]
    params()
    os.unlink(fname)
    print(open('output.evodag').read())
    os.unlink('output.evodag')
    default_nargs()
    del os.environ['KLASS']


def test_model_used_inputs_number():
    import os
    from EvoDAG.command_line import params, train, utils
    fname = mo_training_set()
    sys.argv = ['EvoDAG', '--output-dim=3',
                '-R', '--parameters',
                'cache.evodag', '-p3', '-e1',
                '-r2', fname]
    params()
    sys.argv = ['EvoDAG', '--parameters', 'cache.evodag',
                '-n2', '--output-dim=3',
                '--model', 'model.evodag',
                '--test', fname, fname]
    train()
    sys.argv = ['EvoDAG', '--used-inputs-number', 'model.evodag']
    utils()
    os.unlink('cache.evodag')
    os.unlink('model.evodag')
    default_nargs()


def test_model_min_size():
    import os
    from EvoDAG.command_line import params, train
    import gzip
    import pickle
    fname = mo_training_set()
    sys.argv = ['EvoDAG', '--output-dim=3',
                '-R', '--parameters',
                'cache.evodag', '-p3', '-e2',
                '-r2', fname]
    params()
    sys.argv = ['EvoDAG', '--parameters', 'cache.evodag',
                '-n2', '--output-dim=3',
                '--min-size=6',
                '--model', 'model.evodag',
                '--test', fname, fname]
    train()
    os.unlink('cache.evodag')
    with gzip.open('model.evodag') as fpt:
        m = pickle.load(fpt)
        for x in m.models:
            print(x.size)
            assert x.size >= 6
    os.unlink('model.evodag')
    default_nargs()


def test_logging():
    from EvoDAG.command_line import params
    fname = mo_training_set()
    sys.argv = ['EvoDAG', '--output-dim=3',
                '-R', '--parameters', 'cache.evodag',
                '--verbose=0', '-p3', '-e2', '-r2', fname]
    params()


def test_model_seed():
    import os
    from EvoDAG.command_line import params, train
    fname = mo_training_set()
    sys.argv = ['EvoDAG', '--output-dim=3',
                '-R', '--parameters',
                'cache.evodag', '-p3', '-e2',
                '-r2', fname]
    params()
    sys.argv = ['EvoDAG', '--parameters', 'cache.evodag',
                '-n2', '--output-dim=3', '--seed=1',
                '--model', 'model.evodag',
                '--test', fname, fname]
    c = train(output=True)
    os.unlink('cache.evodag')
    default_nargs()
    assert c.data.seed == 1


def test_create_ensemble():
    import os
    from EvoDAG.command_line import params, train, utils
    from EvoDAG.model import Ensemble
    import pickle
    import gzip
    fname = mo_training_set()
    sys.argv = ['EvoDAG', '--output-dim=3',
                '-R', '--parameters',
                'cache.evodag', '-p3', '-e2',
                '-r2', fname]
    params()
    sys.argv = ['EvoDAG', '--parameters', 'cache.evodag',
                '-n1', '--output-dim=3', '--seed=1',
                '--model', 'model.evodag.1',
                '--test', fname, fname]
    train()
    sys.argv = ['EvoDAG', '--parameters', 'cache.evodag',
                '-n1', '--output-dim=3', '--seed=2',
                '--model', 'model.evodag.2',
                '--test', fname, fname]
    train()
    os.unlink('cache.evodag')
    sys.argv = ['EvoDAG', '--create-ensemble', '-omodel.evodag',
                '-n3', 'model.evodag.* model.evodag.1']
    utils()
    os.unlink('model.evodag.1')
    os.unlink('model.evodag.2')
    with gzip.open('model.evodag', 'r') as fpt:
        ens = pickle.load(fpt)
    assert isinstance(ens, Ensemble)
    os.unlink('model.evodag')
    default_nargs()


def test_input_type_constraint_R():
    from EvoDAG.command_line import params
    from EvoDAG.utils import PARAMS
    fname = mo_training_set()
    sys.argv = ['EvoDAG', '--output-dim=3',
                '-R', '--parameters', 'cache.evodag',
                '--verbose=0', '-p3', '-e2', '-r2', fname]
    c = params(output=True)
    _params = PARAMS.copy()
    c.if_type_contraint(_params)
    print(_params['input_functions'], 'MultipleVariables')
    assert _params['input_functions'][0] == 'MultipleVariables'


def test_input_type_constraint_C():
    from EvoDAG.command_line import params
    from EvoDAG.utils import PARAMS
    fname = mo_training_set()
    sys.argv = ['EvoDAG', '--output-dim=3',
                '-C', '--parameters', 'cache.evodag',
                '--verbose=0', '-p3', '-e2', '-r2', fname]
    c = params(output=True)
    _params = PARAMS.copy()
    c.if_type_contraint(_params)
    print(_params['input_functions'], 'MultipleVariables')
    assert len(_params['input_functions']) == 4
    

def test_params_files():
    import os
    from EvoDAG.command_line import params
    import shutil
    fname = training_set()
    sys.argv = ['EvoDAG', '-C', '-Pcache', '--only-paramsfiles',
                '-r', '2', fname]
    params()
    if os.path.isdir('cache'):
        shutil.rmtree('cache')
        default_nargs()
        return
    assert False


def test_get_best_params_files():
    import numpy as np
    import gzip
    import pickle
    import os
    from EvoDAG.command_line import params, train, utils
    import shutil
    from glob import glob
    if os.path.isdir('cache'):
        shutil.rmtree('cache')
    fname = training_set()
    sys.argv = ['EvoDAG', '-C', '-Pcache', '--only-paramsfiles',
                '-r', '2', fname]
    params()
    for p in glob('cache/*_params.json'):
        basename = p.split('_params.json')[0]
        for s in range(3):
            model = basename + '_%s.model' % s
            sys.argv = ['EvoDAG', '-s%s' % s, '-P%s' % p, '-m%s' % model, '-n1', fname]
            train()
    R = []
    for p in range(2):
        l = []
        for s in range(3):
            with gzip.open('cache/%s_%s.model' % (p, s)) as fpt:
                m = pickle.load(fpt)
                l.append(m.fitness_vs * -1)
        R.append((p, l))
    m = min(R, key=lambda x: np.median(x[1]))
    param = '%s_params.json' % m[0]
    sys.argv = ['EvoDAG', '--best-params-file', 'cache']
    c = utils(output=True)
    assert c.best_params == param
    sys.argv = ['EvoDAG', '-u2', '--best-params-file', 'cache']
    c = utils(output=True)
    assert c.best_params == param
    if os.path.isdir('cache'):
        shutil.rmtree('cache')
        default_nargs()


def test_json2():
    from EvoDAG.command_line import params
    import tempfile
    import json
    fname = tempfile.mktemp()
    with open(fname, 'w') as fpt:
        for x, y in zip(X, cl):
            a = {}
            a['vec'] = [[k, v] for k, v in enumerate(x)]
            a['klass'] = int(y)
            a['vecsize'] = len(x)
            fpt.write(json.dumps(a) + '\n')
    print("termine con el json")
    sys.argv = ['EvoDAG', '-C', '-Poutput.evodag', '--json',
                '-e1', '-p3', '-r2', fname]
    params()
    os.unlink(fname)
    print(open('output.evodag').read())
    os.unlink('output.evodag')
    default_nargs()
