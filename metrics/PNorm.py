from typing import Mapping, Union, Tuple
from gicaf.interface.MetricBase import MetricBase
from numpy.linalg import norm
from numpy import ravel, inf, ndarray

class AbsValueNorm(MetricBase):

    def __call__(
        self, 
        image: ndarray, 
        adversarial_image: ndarray, 
        model_metadata: Mapping[str, Union[int, bool, Tuple[int, int]]]
    ) -> float: 
        img = image
        adv = adversarial_image
        for _ in range(model_metadata['channels'] - 1):
            img = ravel(img)
            adv = ravel(adv)
        return norm(adv - img, 0)

class EuclideanNorm(MetricBase):

    def __call__(
        self, 
        image: ndarray, 
        adversarial_image: ndarray, 
        model_metadata: Mapping[str, Union[int, bool, Tuple[int, int]]]
    ) -> float: 
        return norm(adversarial_image - image, 2)

class InfNorm(MetricBase):

    def __call__(
        self, 
        image: ndarray, 
        adversarial_image: ndarray, 
        model_metadata: Mapping[str, Union[int, bool, Tuple[int, int]]]
    ) -> float: 
        return norm(adversarial_image - image, inf)
