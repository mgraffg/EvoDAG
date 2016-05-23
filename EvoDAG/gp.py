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


from .node import Function
from .model import Model


class Individual(object):
    """Object to store in individual on prefix notation"""

    def __init__(self, ind, classifier=True, labels=None):
        self._ind = ind
        self._pos = 0
        self._classifier = classifier
        self._labels = labels
        self._X = None

    @property
    def individual(self):
        "Individual"
        return self._ind

    @individual.setter
    def individual(self, v):
        self._ind = v

    def decision_function(self, X):
        "Decision function i.e. the raw data of the prediction"
        if X is None:
            if self._classifier:
                return self._hy_test.boundaries()
            return self._hy_test
        self._X = Model.convert_features(X)
        self._eval()
        return self._ind[0].hy

    def _eval(self):
        "Evaluates a individual using recursion and self._pos as pointer"
        pos = self._pos
        self._pos += 1
        node = self._ind[pos]
        if isinstance(node, Function):
            args = [self._eval() for x in range(node.nargs)]
            node.eval(args)
        else:
            node.eval(self._X)
        return node
