from typing import Mapping, Union, Tuple
from gicaf.interface.MetricBase import MetricBase
from skimage.metrics import structural_similarity
from numpy import ndarray

class SSIM(MetricBase):

    def __call__(
        self, 
        image: ndarray, 
        adversarial_image: ndarray, 
        model_metadata: Mapping[str, Union[int, bool, Tuple[int, int]]]
    ) -> float: 
        return structural_similarity(image, 
                                adversarial_image, 
                                multichannel=True if model_metadata['channels'] > 1 else False)

class PASS(MetricBase):

    def __call__(
        self, 
        image: ndarray, 
        adversarial_image: ndarray, 
        model_metadata: Mapping[str, Union[int, bool, Tuple[int, int]]]
    ) -> float: 
        return structural_similarity(image, 
                                adversarial_image, 
                                multichannel=True if model_metadata['channels'] > 1 else False)
