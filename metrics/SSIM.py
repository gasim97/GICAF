from gicaf.interface.MetricInterface import MetricInterface
from skimage.metrics import structural_similarity

class SSIM(MetricInterface):

    def __call__(self, image, adversarial_image, model_metadata): 
        return structural_similarity(image, 
                                adversarial_image, 
                                multichannel=True if model_metadata['channels'] > 1 else False)
