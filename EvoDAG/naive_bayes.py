# Copyright 2017 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from .cython_utils import naive_bayes_mean_std2, naive_bayes_Nc
import numpy as np


class NaiveBayes(object):
    """Computes the class statistics used by the naive Bayes nodes from the training examples selected by the mask."""
    def __init__(self, mask=None, nclass=None, klass=None):
        self._mask = np.flatnonzero(mask).astype(np.uint32)
        self._nclass = nclass
        self._klass = np.asarray(klass).astype(np.uint32)

    def coef(self, var):
        """Return the mean, variance and prior of each class for the variable var."""
        return naive_bayes_mean_std2(var, self._klass, self._mask, self._nclass)

    def coef_MN(self, var):
        """Return the log proportion of the mass of the variable var in each class and the log prior of each class."""
        return naive_bayes_Nc(var, self._klass, self._mask, self._nclass)


