from gicaf.interface.MetricBase import MetricBase
from skimage.metrics import peak_signal_noise_ratio

class PSNR(MetricBase):

    def __call__(self, image, adversarial_image, model_metadata): 
        return peak_signal_noise_ratio(image,
                                    adversarial_image, 
                                    data_range=(model_metadata['bounds'][1] - model_metadata['bounds'][0]))
