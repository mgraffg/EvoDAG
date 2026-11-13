# Copyright 2013 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from Cython.Build import cythonize
from setuptools import Extension, setup
import numpy as np

extension = [Extension('EvoDAG.linalg_solve', ["EvoDAG/linalg_solve.pyx"],
                       include_dirs=[np.get_include()]),
             Extension('EvoDAG.function_selection', ["EvoDAG/function_selection.pyx"],
                       include_dirs=[np.get_include()]),
             Extension('EvoDAG.cython_utils', ["EvoDAG/cython_utils.pyx"],
                       include_dirs=[np.get_include()])]

# Everything else (metadata, version, packages) lives in pyproject.toml;
# setuptools cannot declare Cython extensions there.
setup(ext_modules=cythonize(extension,
                            compiler_directives={'language_level': 3,
                                                 'profile': False,
                                                 'nonecheck': False,
                                                 'boundscheck': False}))
