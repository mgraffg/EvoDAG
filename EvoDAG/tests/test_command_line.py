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
from EvoDAG.command_line import CommandLine, main
from test_root import X, cl


def training_set():
    import tempfile
    fname = tempfile.mktemp()
    with open(fname, 'x') as fpt:
        for x, v in zip(X, cl):
            l = x.tolist()
            l.append(v)
            fpt.write(','.join(map(str, l)))
            fpt.write('\n')
    return fname


def test_command_line():
    fname = training_set()
    sys.argv = ['EvoDAG', '-s', '1', '-e', '10', '-p', '100', fname]
    c = CommandLine()
    c.parse_args()
    assert c.data.training_set == fname
    os.unlink(fname)
    os.unlink(c.data.output_file)
    assert c.evo._early_stopping_rounds == 10
    assert c.evo._classifier
    assert c.evo._popsize == 100
    assert c.evo._seed == 1


def test_main():
    fname = training_set()
    sys.argv = ['EvoDAG', '-o', 'temp.evodag.gz',
                '-e', '10', '-p', '100', fname, '-t', fname]
    main()
    os.unlink(fname)
    os.unlink('temp.evodag.gz')
    os.unlink(fname + '.evodag.csv')


def test_optimize_parameters():
    import os
    fname = training_set()
    sys.argv = ['EvoDAG', '--cache-file',
                'cache.evodag.gz', '-p10', '-e2', '-r', '2', fname]
    c = CommandLine()
    c.parse_args()
    os.unlink(fname)
    os.unlink(c.data.output_file)
    assert c.evo._popsize == 10
    assert os.path.isfile('cache.evodag.gz')
    os.unlink('cache.evodag.gz')


def test_previous_model():
    import gzip
    import pickle
    fname = training_set()
    sys.argv = ['EvoDAG', '-p10', '-e2', '-omodel.evodag.gz', fname]
    c = CommandLine()
    c.parse_args()
    with gzip.open(c.data.output_file, 'w') as fpt:
        pickle.dump([], fpt)
    c = CommandLine()
    c.parse_args()
    os.unlink(fname)
    os.unlink(c.data.output_file)
    assert isinstance(c.model, list) and len(c.model) == 0


def test_cpu_cores():
    import os
    fname = training_set()
    sys.argv = ['EvoDAG', '-u2', '--cache-file',
                'cache.evodag.gz', '-p10', '-e2', '-r', '2', fname]
    c = CommandLine()
    c.parse_args()
    os.unlink(fname)
    os.unlink(c.data.output_file)
    assert c.evo._popsize == 10
    assert os.path.isfile('cache.evodag.gz')
    os.unlink('cache.evodag.gz')
