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
import array


class NaiveBayes(object):
    def __init__(self, mask=None, nclass=None, klass=None):
        self._mask = mask.index
        self._nclass = nclass
        self._klass = array.array('I', [int(x) for x in klass.full_array()])

    def coef(self, var):
        return naive_bayes_mean_std2(var, self._klass, self._mask, self._nclass)

    def coef_MN(self, var):
        return naive_bayes_Nc(var, self._klass, self._mask, self._nclass)


