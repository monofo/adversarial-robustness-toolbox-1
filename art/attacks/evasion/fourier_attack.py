# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the single Fourier attack.

| Paper link: https://arxiv.org/abs/1809.04098
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Union

import numpy as np
from scipy.fft import ifftn

from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import (
    ClassGradientsMixin,
    ClassifierGradients,
)
from art.config import ART_NUMPY_DTYPE
from art.utils import projection

logger = logging.getLogger(__name__)


class FourierAttack(EvasionAttack):
    attack_params = EvasionAttack.attack_params + [
        'epsilon',
        'block_size',
        'eps',
        'norm',
        'batch_size',
    ]

    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)

    def __init__(
        self,
        classifier: ClassifierGradients,
        epsilon: float = 20.0,
        block_size: int = 4,
        eps: float = 0.1,
        norm: Union[int, float, str] = np.inf,
        batch_size: int = 1,
    ):
        """
        Create a single Fourier attack instance.

        :param epsilon: overshoot parameter
        :param block_size: block size for Fourier attacks.
        :param eps: noise size
        :param batch_size: Internal size of batches for prediction.
        :param norm: The norm of the adversarial perturbation. Possible values: "inf", np.inf, 2
        """
        super(FourierAttack, self).__init__(estimator=classifier)
        self.epsilon = epsilon
        self.block_size = block_size
        self.eps = eps
        self.norm = norm
        self.batch_size = batch_size
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :param y: An array with the original labels to be predicted.
        :return: An array holding the adversarial examples.
        """
        x = x.astype(ART_NUMPY_DTYPE)

        if self.estimator.channels_first:
            nb_channels = x.shape[1]
            nb_xdim = x.shape[2]
            nb_ydim = x.shape[3]
        else:
            nb_channels = x.shape[3]
            nb_xdim = x.shape[1]
            nb_ydim = x.shape[2]

        if nb_xdim != nb_ydim:
            raise ValueError('Input images must be square.')

        clip_min = -np.inf
        clip_max = np.inf
        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values

        # Init
        noise = np.zeros((1, nb_xdim, nb_ydim, nb_channels))
        if self.estimator.channels_first:
            noise = noise.transpose(0, 3, 1, 2)
        fooling_rate = 0.0
        max_fooling_rate = 0.0
        nb_instances = len(x)

        # get the labels
        if y is None:
            # use the predicted labels
            logger.info("Using model predictions as the correct labels.")
            pred_y = self.estimator.predict(x, batch_size=self.batch_size)
        else:
            # use the actual labels
            pred_y = y
        
        correct_y_max = np.argmax(pred_y, axis=1)

        nb_blocks = int(nb_xdim / self.block_size)
        for i in range(nb_blocks):
            for j in range(nb_blocks):
                # get Fourier basis
                xf = np.zeros((nb_xdim, nb_ydim))
                xf[i * self.block_size, j * self.block_size] = 1.0
                Z = ifftn(xf)
                uap_sfa_1 = np.real(Z)

                xf = np.zeros((nb_xdim, nb_ydim))
                xf[nb_xdim - i * self.block_size - 1, nb_ydim - j * self.block_size - 1] = 1.0
                Z = ifftn(xf)
                uap_sfa_2 = np.real(Z)

                uap_sfa = (1 + i) * uap_sfa_1 + (1 - i) * uap_sfa_2

                # generate noise
                tmp_noise = np.zeros((1, nb_xdim, nb_ydim, nb_channels))
                for c in range(nb_channels):
                    tmp_noise[:, :, :, c] = tmp_noise[:, :, :, c] + self.epsilon * uap_sfa
                
                # projection
                tmp_noise = projection(tmp_noise, self.eps, self.norm)
                if self.estimator.channels_first:
                    tmp_noise = tmp_noise.transpose(0, 3, 1, 2)

                # Apply attack and clip
                x_adv = x + tmp_noise
                if self.estimator.clip_values is not None:
                    x_adv = np.clip(x_adv, clip_min, clip_max)

                # Compute the fooling rate
                y_adv = np.argmax(self.estimator.predict(x_adv, batch_size=self.batch_size), axis=1)
                fooling_rate = np.sum(correct_y_max != y_adv) / nb_instances

                if max_fooling_rate < fooling_rate:
                    max_fooling_rate = fooling_rate
                    noise = tmp_noise

            val_norm = np.linalg.norm(noise.flatten(), ord=self.norm)
            logger.info('Success rate of Fourier attack at section %d: %.2f%% (L%s norm of noise: %.2f)', i, 100 * max_fooling_rate, str(self.norm), val_norm)

        self.fooling_rate = max_fooling_rate
        self.noise = noise
        logger.info("Final success rate of Fourier attack: %.2f%%", 100 * max_fooling_rate)

        # generate adversarial examples
        x_adv = x + noise

        return x_adv


    def _check_params(self) -> None:

        if self.epsilon < 0:
            raise ValueError("The overshoot parameter must not be negative.")

        if self.block_size <= 0:
            raise ValueError('The block size `block_size` has to be positive.')

        if not isinstance(self.eps, (float, int)) or self.eps <= 0:
            raise ValueError("The eps coefficient must be a positive float.")
        
        if self.batch_size <= 0:
            raise ValueError('The batch size `batch_size` has to be positive.')

