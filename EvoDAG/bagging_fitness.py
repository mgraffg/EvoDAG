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
from .cython_utils import fitness_SAE, Score


class BaggingFitness(object):
    def __init__(self, base=None):
        self._base = base
        self.assert_fitness_function()

    def assert_fitness_function(self):
        base = self._base
        assert base._fitness_function in ['ER', 'macro-F1', 'BER', 'macro-Precision',
                                          'DotF1', 'DotRecall', 'DotPrecision',
                                          'DotRecallDotPrecision', 'F1',
                                          'RecallDotPrecision', 'macro-RecallF1',
                                          'accDotMacroF1',
                                          'accuracy', 'g_recall_precision',
                                          'g_precision', 'a_precision', 'g_recall',
                                          'g_g_recall_precision', 'g_F1', 'a_F1']

    @property
    def nclasses(self):
        try:
            return self._nclasses
        except AttributeError:
            self._nclasses = self._base._labels.shape[0]
        return self._nclasses

    @property
    def score(self):
        try:
            return self._score
        except AttributeError:
            self._score = Score(self.nclasses)
        return self._score

    @property
    def min_class(self):
        try:
            return self._min_class
        except AttributeError:
            klass = tonparray(self._base._y_klass)
            c = [(klass == x).sum() for x in np.unique(klass)]
            self._min_class = np.argmin(c)
        return self._min_class

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
        elif base._fitness_function == 'macro-RecallF1':
            return self.mask_fitness_BER(k)
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

    def del_error(self, v):
        try:
            delattr(v, '_error')
        except AttributeError:
            pass

    def fitness(self, v):
        "Fitness function in the training set"
        base = self._base
        if base._classifier:
            if base._multiple_outputs:
                hy = SparseArray.argmax(v.hy)
                fit_func = base._fitness_function
                if fit_func == 'macro-F1' or fit_func == 'a_F1':
                    f1_score = self.score
                    mf1, mf1_v = f1_score.a_F1(base._y_klass, hy, base._mask_ts.index)
                    v._error = mf1_v - 1
                    v.fitness = mf1 - 1
                elif fit_func == 'DotF1' or fit_func == 'g_F1':
                    f1_score = self.score
                    mf1, mf1_v = f1_score.g_F1(base._y_klass, hy, base._mask_ts.index)
                    v._error = mf1_v - 1
                    v.fitness = mf1 - 1
                elif fit_func == 'DotRecallDotPrecision' or fit_func == 'g_g_recall_precision':
                    f1_score = self.score
                    mf1, mf1_v = f1_score.g_g_recall_precision(base._y_klass, hy,
                                                               base._mask_ts.index)
                    v._error = mf1_v - 1
                    v.fitness = mf1 - 1
                elif fit_func == 'BER' or fit_func == 'a_recall':
                    f1_score = self.score
                    mf1, mf1_v = f1_score.a_recall(base._y_klass, hy, base._mask_ts.index)
                    v._error = mf1_v - 1
                    v.fitness = mf1 - 1
                elif fit_func == 'DotRecall' or fit_func == 'g_recall':
                    f1_score = self.score
                    mf1, mf1_v = f1_score.g_recall(base._y_klass, hy,
                                                   base._mask_ts.index)
                    v._error = mf1_v - 1
                    v.fitness = mf1 - 1
                elif fit_func == 'macro-Precision' or fit_func == 'a_precision':
                    f1_score = self.score
                    mf1, mf1_v = f1_score.a_precision(base._y_klass, hy,
                                                      base._mask_ts.index)
                    v._error = mf1_v - 1
                    v.fitness = mf1 - 1
                elif fit_func == 'DotPrecision' or fit_func == 'g_precision':
                    f1_score = self.score
                    mf1, mf1_v = f1_score.g_precision(base._y_klass, hy,
                                                      base._mask_ts.index)
                    v._error = mf1_v - 1
                    v.fitness = mf1 - 1
                elif fit_func == 'accDotMacroF1':
                    f1_score = self.score
                    mf1, mf1_v = f1_score.accDotMacroF1(base._y_klass, hy,
                                                        base._mask_ts.index)
                    v._error = mf1_v - 1
                    v.fitness = mf1 - 1
                elif fit_func == 'macro-RecallF1':
                    f1_score = self.score
                    mf1, mf1_v = f1_score.macroRecallF1(base._y_klass, hy,
                                                        base._mask_ts.index)
                    v._error = mf1_v - 1
                    v.fitness = mf1 - 1
                elif fit_func == 'F1':
                    f1_score = self.score
                    mf1, mf1_v = f1_score.F1(self.min_class, base._y_klass,
                                             hy, base._mask_ts.index)
                    v._error = mf1_v - 1
                    v.fitness = mf1 - 1
                elif fit_func == 'RecallDotPrecision' or fit_func == 'g_recall_precision':
                    f1_score = self.score
                    mf1, mf1_v = f1_score.g_recall_precision(self.min_class,
                                                             base._y_klass,
                                                             hy, base._mask_ts.index)
                    v._error = mf1_v - 1
                    v.fitness = mf1 - 1
                elif fit_func == 'ER' or fit_func == 'accuracy':
                    f1_score = self.score
                    mf1, mf1_v = f1_score.accuracy(base._y_klass,
                                                   hy, base._mask_ts.index)
                    v._error = mf1_v - 1
                    v.fitness = mf1 - 1
                else:
                    raise RuntimeError('Unknown fitness function %s' % base._fitness_function)
            else:
                v.fitness = -base._ytr.SSE(v.hy * base._mask)
        else:
            if base._multiple_outputs:
                v.fitness = fitness_SAE(base._ytr, v.hy, base._mask)
            else:
                v.fitness = -base._ytr.SAE(v.hy * base._mask)

    def fitness_vs(self, v):
        """Fitness function in the validation set
        In classification it uses BER and RSE in regression"""
        base = self._base
        if base._classifier:
            if base._multiple_outputs:
                v.fitness_vs = v._error
                # if base._fitness_function == 'macro-F1':
                #     v.fitness_vs = v._error
                # elif base._fitness_function == 'BER':
                #     v.fitness_vs = v._error
                # elif base._fitness_function == 'macro-Precision':
                #     v.fitness_vs = v._error
                # elif base._fitness_function == 'accDotMacroF1':
                #     v.fitness_vs = v._error
                # elif base._fitness_function == 'macro-RecallF1':
                #     v.fitness_vs = v._error
                # elif base._fitness_function == 'F1':
                #     v.fitness_vs = v._error
                # else:
                #     v.fitness_vs = - v._error.dot(base._mask_vs) / base._mask_vs.sum()
            else:
                v.fitness_vs = -((base.y - v.hy.sign()).sign().fabs() *
                                 base._mask_vs).sum()
        else:
            mask = base._mask
            y = base.y
            hy = v.hy
            if not isinstance(mask, list):
                mask = [mask]
                y = [y]
                hy = [hy]
            fit = []
            for _mask, _y, _hy in zip(mask, y, hy):
                m = (_mask + -1).fabs()
                x = _y * m
                y = _hy * m
                a = (x - y).sq().sum()
                b = (x + -x.sum() / x.size()).sq().sum()
                fit.append(-a / b)
            v.fitness_vs = np.mean(fit)

    def set_fitness(self, v):
        """Set the fitness to a new node.
        Returns false in case fitness is not finite"""
        base = self._base
        self.fitness(v)
        if not np.isfinite(v.fitness):
            self.del_error(v)
            return False
        if base._tr_fraction < 1:
            self.fitness_vs(v)
            if not np.isfinite(v.fitness_vs):
                self.del_error(v)
                return False
        self.del_error(v)
        return True
        
