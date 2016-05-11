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


import numpy as np


def test_parameter_grid_results():
    from EvoDAG.utils import RandomParameterSearch
    a = RandomParameterSearch()
    assert len([x for x in a]) == 1468


def test_process_params():
    from EvoDAG.utils import RandomParameterSearch
    from EvoDAG import EvoDAG
    rs = RandomParameterSearch(npoints=1)
    args = [x for x in rs][0]
    evo = EvoDAG(**rs.process_params(args))
    params = evo.get_params()
    for k, v in args.items():
        if k in params:
            assert v == params[k]


def test_random_parameter_search_len():
    from itertools import product
    from EvoDAG.utils import RandomParameterSearch
    params = {'a': [1, 2, 3],
              'b': [-1, -2],
              'c': ['a', 'b', 'c', 'd', 'e']}
    rs = RandomParameterSearch(params=params)
    params = sorted(params.items())
    a = product(*[x[1] for x in params])
    assert len([x for x in a]) == len(rs)


def test_random_parameter_search_getitem():
    from itertools import product
    from EvoDAG.utils import RandomParameterSearch
    params = {'a': [1, 2, 3],
              'b': [-1, -2],
              'c': ['a', 'b', 'c', 'd', 'e']}
    rs = RandomParameterSearch(params=params)
    params = sorted(params.items())
    a = product(*[x[1] for x in params])
    for k, v in enumerate(a):
        v1 = [x[1] for x in sorted(rs[k].items())]
        print(v, v1)
        assert np.all([x == y for x, y in zip(v, v1)])

    
