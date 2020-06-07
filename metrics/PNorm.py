from gicaf.interface.MetricInterface import MetricInterface
from numpy.linalg import norm
from numpy import ravel, inf

class AbsValueNorm(MetricInterface):

    def __call__(self, image, adversarial_image, model_metadata): 
        img = image
        adv = adversarial_image
        for _ in range(model_metadata['channels'] - 1):
            img = ravel(img)
            adv = ravel(adv)
        return norm(adv - img, 0)

class EuclideanNorm(MetricInterface):

    def __call__(self, image, adversarial_image, model_metadata): 
        return norm(adversarial_image - image, 2)

class InfNorm(MetricInterface):

    def __call__(self, image, adversarial_image, model_metadata): 
        return norm(adversarial_image - image, inf)
