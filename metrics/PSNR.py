from typing import Mapping, Union, Tuple
from gicaf.interface.MetricBase import MetricBase
from skimage.metrics import peak_signal_noise_ratio
from numpy import ndarray

class PSNR(MetricBase):

    def __call__(
        self, 
        image: ndarray, 
        adversarial_image: ndarray, 
        model_metadata: Mapping[str, Union[int, bool, Tuple[int, int]]]
    ) -> float: 
        return peak_signal_noise_ratio(image,
                                    adversarial_image, 
                                    data_range=(model_metadata['bounds'][1] - model_metadata['bounds'][0]))
