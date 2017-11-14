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
from setuptools import setup
from Cython.Build import cythonize
from setuptools import Extension
from os.path import join
import numpy as np

with open('README.rst') as fpt:
    long_desc = fpt.read()
version = open("VERSION").readline().lstrip().rstrip()
lst = open(join("EvoDAG", "__init__.py")).readlines()
for k in range(len(lst)):
    v = lst[k]
    if v.count("__version__"):
        lst[k] = "__version__ = '%s'\n" % version
with open(join("EvoDAG", "__init__.py"), "w") as fpt:
    fpt.write("".join(lst))

extension = [Extension('EvoDAG.linalg_solve', ["EvoDAG/linalg_solve.pyx"],
                       include_dirs=[np.get_include()]),
             Extension('EvoDAG.function_selection', ["EvoDAG/function_selection.pyx"],
                       include_dirs=[np.get_include()]),
             Extension('EvoDAG.cython_utils', ["EvoDAG/cython_utils.pyx"],
                       include_dirs=[np.get_include()])]
    
setup(
    name="EvoDAG",
    description="""Evolving Directed Acyclic Graph""",
    long_description=long_desc,
    version=version,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        "Topic :: Scientific/Engineering :: Artificial Intelligence"],
    url='https://github.com/mgraffg/EvoDAG',
    author="Mario Graff",
    author_email="mgraffg@ieee.org",
    packages=['EvoDAG', 'EvoDAG/tests'],
    include_package_data=True,
    zip_safe=False,
    ext_modules=cythonize(extension,
                          compiler_directives={'profile': False,
                                               'nonecheck': False,
                                               'boundscheck': False}),

    package_data={'EvoDAG/conf': ['parameter_values.json',
                                  'default_parameters.json',
                                  'default_parameters_r.json'],
                  '': ['*.pxd']},
    install_requires=['numpy', 'SparseArray'],
    entry_points={
        'console_scripts': ['EvoDAG-params=EvoDAG.command_line:params',
                            'EvoDAG-train=EvoDAG.command_line:train',
                            'EvoDAG-predict=EvoDAG.command_line:predict',
                            'EvoDAG-utils=EvoDAG.command_line:utils'],
    }
)
