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


def test_parameter_grid_results():
    from EvoDAG.utils import parameter_grid_results
    a = parameter_grid_results()
    assert len(a) == 1468


def test_process_params():
    from EvoDAG.utils import parameter_grid_results, process_params
    from EvoDAG import EvoDAG
    args = parameter_grid_results()[0]
    evo = EvoDAG(**process_params(args))
    params = evo.get_params()
    for k, v in args.items():
        if k in params:
            assert v == params[k]
