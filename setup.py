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
import numpy
from setuptools import setup
from setuptools import Extension
from Cython.Distutils import build_ext
from distutils import sysconfig
from distutils.command.clean import clean
import os
from os.path import join


class Clean(clean):
    def run(self):
        clean.run(self)
        if self.all:
            for dir, dirs, files in os.walk('EvoDAG'):
                abspath = os.path.abspath(dir)
                for f in files:
                    ext = f.split('.')[-1]
                    if ext in ['c', 'so']:
                        os.unlink(os.path.join(abspath, f))


# -mno-fused-madd

lst = ['CFLAGS', 'CONFIG_ARGS', 'LIBTOOL', 'PY_CFLAGS']
for k, v in zip(lst, sysconfig.get_config_vars(*lst)):
    if v is None:
        continue
    v = v.replace('-mno-fused-madd', '')
    os.environ[k] = v
ext_modules = [Extension("EvoDAG.sparse_array",
                         [join("EvoDAG", "sparse_array.pxd"),
                          join("EvoDAG", "sparse_array.pyx")],
                         # libraries=["m"],
                         include_dirs=[numpy.get_include()])]

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
    cmdclass={"build_ext": build_ext, "clean": Clean},
    ext_modules=ext_modules,
    packages=['EvoDAG', 'EvoDAG/tests'],
    include_package_data=True,
    zip_safe=False,
    package_data={'': ['*.pxd']},
    install_requires=['cython >= 0.19.2', 'numpy >= 1.6.2']
)
