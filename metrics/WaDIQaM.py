from gicaf.interface.MetricInterface import MetricInterface
from gicaf.metrics.resources.wadiqam import Utils
from cv2 import cvtColor, COLOR_BGR2RGB

class WaDIQaM(MetricInterface):

    def __call__(self, image, adversarial_image, model_metadata): 
        data = Utils.NonOverlappingCropPatches(cvtColor(adversarial_image, COLOR_BGR2RGB) if model_metadata['bgr'] else adversarial_image,
                                            cvtColor(image, COLOR_BGR2RGB) if model_metadata['bgr'] else image)
        model = Utils.get_FRnet()
        dist_patches = data[0].unsqueeze(0)
        ref_patches = data[1].unsqueeze(0)
        score = model((dist_patches, ref_patches))
        return score.item()