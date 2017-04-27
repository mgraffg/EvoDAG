# Copyright 2015 Mario Graff Guerrero

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
from SparseArray import SparseArray
from .utils import tonparray


class BaggingFitness(object):
    def __init__(self, base=None):
        self._base = base

    def mask_fitness_BER(self, k):
        base = self._base
        k = k.argmax(axis=1)
        base._y_klass = SparseArray.fromlist(k)
        klass = np.unique(k)
        cnt = np.min([(k == x).sum() for x in klass]) * (1 - base._tr_fraction)
        cnt = int(np.floor(cnt))
        if cnt == 0:
            cnt = 1
        mask = np.ones_like(k, dtype=np.bool)
        mask_ts = np.zeros(k.shape[0])
        for i in klass:
            index = np.where(k == i)[0]
            np.random.shuffle(index)
            mask[index[:cnt]] = False
            mask_ts[index[cnt:]] = 1.0 / (1.0 * index[cnt:].shape[0] * klass.shape[0])
        base._mask_vs = SparseArray.fromlist(~mask)
        base._mask_ts = SparseArray.fromlist(mask_ts)
        return mask

    def mask_fitness_function(self, k):
        base = self._base
        if base._fitness_function == 'BER':
            return self.mask_fitness_BER(k)
        elif base._fitness_function == 'ER':
            k = k.argmax(axis=1)
            base._y_klass = SparseArray.fromlist(k)
            cnt = k.shape[0] * (1 - base._tr_fraction)
            cnt = int(np.floor(cnt))
            if cnt == 0:
                cnt = 1
            mask = np.ones_like(k, dtype=np.bool)
            mask_ts = np.zeros(k.shape[0])
            index = np.arange(k.shape[0])
            np.random.shuffle(index)
            mask[index[:cnt]] = False
            mask_ts[index[cnt:]] = 1.0
            base._mask_vs = SparseArray.fromlist(~mask)
            base._mask_ts = SparseArray.fromlist(mask_ts / mask_ts.sum())
            return mask
        raise RuntimeError('Unknown fitness function %s' % self._fitness_function)

    def set_classifier_mask(self, v, base_mask=True):
        """Computes the mask used to create the training and validation set"""
        base = self._base
        v = tonparray(v)
        a = np.unique(v)
        if a[0] != -1 or a[1] != 1:
            raise RuntimeError("The labels must be -1 and 1 (%s)" % a)
        mask = np.zeros_like(v)
        cnt = min([(v == x).sum() for x in a]) * base._tr_fraction
        cnt = int(round(cnt))
        for i in a:
            index = np.where((v == i) & base_mask)[0]
            np.random.shuffle(index)
            mask[index[:cnt]] = True
        base._mask = SparseArray.fromlist(mask)
        return SparseArray.fromlist(v)

    def transform_to_mo(self, v):
        base = self._base
        klass = base._labels
        y = np.empty((v.shape[0], klass.shape[0]))
        y.fill(-1)
        for i, k in enumerate(klass):
            mask = k == v
            y[mask, i] = 1
        return y

    def multiple_outputs_cl(self, v):
        base = self._base
        if isinstance(v, list):
            assert len(v) == base._labels.shape[0]
            v = np.array([tonparray(x) for x in v]).T
        else:
            v = tonparray(v)
            v = self.transform_to_mo(v)
        base_mask = self.mask_fitness_function(v)
        mask = []
        ytr = []
        y = []
        for _v in v.T:
            _v = SparseArray.fromlist(_v)
            self.set_classifier_mask(_v, base_mask)
            mask.append(base._mask)
            ytr.append(_v * base._mask)
            y.append(_v)
            base._y = _v
        base._ytr = ytr
        base._y = y
        base._mask = mask

    def set_regression_mask(self, v):
        """Computes the mask used to create the training and validation set"""
        base = self._base
        index = np.arange(v.size())
        np.random.shuffle(index)
        ones = np.ones(v.size())
        ones[index[int(base._tr_fraction * v.size()):]] = 0
        base._mask = SparseArray.fromlist(ones)

    def test_regression_mask(self, v):
        """Test whether the average prediction is different than zero"""
        base = self._base
        m = (base._mask + -1.0).fabs()
        x = v * m
        b = (x + -x.sum() / x.size()).sq().sum()
        return b != 0

    def multiple_outputs_regression(self, v):
        base = self._base
        assert isinstance(v, list)
        v = np.array([tonparray(x) for x in v]).T
        mask = []
        ytr = []
        y = []
        for _v in v.T:
            _v = SparseArray.fromlist(_v)
            for _ in range(base._number_tries_feasible_ind):
                self.set_regression_mask(_v)
                flag = self.test_regression_mask(_v)
                if flag:
                    break
            if not flag:
                msg = "Unsuitable validation set (RSE: average equals zero)"
                raise RuntimeError(msg)
            mask.append(base._mask)
            ytr.append(_v * base._mask)
            y.append(_v)
            base._y = _v
        base._ytr = ytr
        base._y = y
        base._mask = mask

    def mask_vs(self):
        """Procedure to perform, in classification,
        more efficiently BER in the validation set"""
        base = self._base
        if not base._classifier:
            return
        if base._tr_fraction == 1:
            return
        m = ~ tonparray(base._mask).astype(np.bool)
        f = np.zeros(len(base._mask))
        y = tonparray(base.y)
        den = (y[m] == -1).sum()
        if den:
            f[y == -1] = 0.5 / den
        else:
            f[y == -1] = 0.5
        den = (y[m] == 1).sum()
        if den:
            f[y == 1] = 0.5 / den
        else:
            f[y == 1] = 0.5
        f[~m] = 0
        base._mask_vs = SparseArray.fromlist(f)
