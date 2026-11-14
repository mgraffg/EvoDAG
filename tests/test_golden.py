# Copyright 2024 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compares the library with the golden outputs in ``golden/``.

The cases and the way they are compared are defined in ``golden/record.py``,
which also regenerates the golden files. Structure (class names, variables,
which evaluations succeed) must be identical, numbers must agree within
``1e-6`` (absolute, plus ``1e-9`` relative for huge values) and non finite
values must be in the same places.

The groups that are a whole evolutionary run (``record.PLATFORM_GROUPS``) are
reproducible only where floating point arithmetic gives the same bits as on
the machine that recorded them; elsewhere they are skipped.
"""
import inspect
import random

import numpy as np
import pytest

import EvoDAG.node as node_module
import EvoDAG.cython_utils as cython_utils
import EvoDAG.linalg_solve as linalg_solve
from golden import record


MISSING = object()


@pytest.fixture(autouse=True)
def preserve_global_state():
    """The cases change node classes' ``nargs`` (also creating the attribute
    in the classes that inherit it) and the random generators: leave them as
    they were found, so the order of the test files does not matter."""
    classes = [getattr(node_module, name) for name in record.PRISTINE_NARGS]
    nargs = {c: vars(c).get('nargs', MISSING) for c in classes}
    np_state = np.random.get_state()
    py_state = random.getstate()
    yield
    for cls, value in nargs.items():
        if value is MISSING:
            if 'nargs' in vars(cls):
                del cls.nargs
        else:
            cls.nargs = value
    np.random.set_state(np_state)
    random.setstate(py_state)


@pytest.mark.parametrize('group', list(record.GROUPS))
def test_golden(group):
    if (group in record.PLATFORM_GROUPS and
            record.fingerprint() != record.recorded_fingerprint()):
        pytest.skip('an evolutionary run is reproducible only on a platform '
                    'with the arithmetic of the one that recorded it')
    problems = record.compare(group)
    assert not problems, '\n'.join(problems[:20])


def golden_keys(group):
    arrays, meta = record.load(group)
    return set(arrays) | set(meta)


def covered(keys, name):
    return any(k == name or k.startswith(name + '.') for k in keys)


def test_every_node_class_is_covered():
    keys = golden_keys('nodes_single') | golden_keys('nodes_multiple')
    abstract = {'Function', 'Function1', 'OneArg'}
    classes = [c for _, c in inspect.getmembers(node_module, inspect.isclass)
               if c.__module__ == node_module.__name__ and
               issubclass(c, node_module.Variable) and
               c.__name__ not in abstract]
    assert classes
    missing = [c.__name__ for c in classes if not covered(keys, c.__name__)]
    assert not missing
    for cls in record.NODE_CLASSES:
        assert cls in classes


def public_names(module):
    return [name for name, obj in vars(module).items()
            if not name.startswith('_') and
            callable(obj) and getattr(obj, '__module__', None) == module.__name__]


def test_every_kernel_is_covered():
    keys = golden_keys('kernels')
    names = public_names(cython_utils) + public_names(linalg_solve)
    assert 'compute_weight' in names and 'fitness_SSE' in names
    assert 'Score' in names and 'centroid_coef' in names
    missing = [name for name in names if not covered(keys, name)]
    assert not missing
    methods = [name for name, obj in vars(cython_utils.Score).items()
               if not name.startswith('_') and callable(obj)]
    assert 'F1' in methods and 'accDotMacroF1' in methods
    missing = [m for m in methods
               if not any(('.%s' % m) in k or k.endswith(m) for k in keys)]
    assert not missing


def test_fingerprint_is_stable_and_platform_groups_exist():
    assert record.fingerprint() == record.fingerprint()
    assert len(record.recorded_fingerprint()) == 64
    assert 'endtoend_classifier' in record.PLATFORM_GROUPS
    assert 'density' in record.PLATFORM_GROUPS
    assert not {'nodes_single', 'nodes_multiple', 'kernels', 'masks',
                'readers'} & record.PLATFORM_GROUPS
