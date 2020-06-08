from gicaf.interface.MetricBase import MetricBase
from gicaf.metrics.resources.wadiqam import Utils
from cv2 import cvtColor, COLOR_BGR2RGB

class WaDIQaM(MetricBase):

    def __init__(self):
        self.model = Utils.get_FRnet()

    def __call__(self, image, adversarial_image, model_metadata): 
        data = Utils.NonOverlappingCropPatches(cvtColor(adversarial_image, COLOR_BGR2RGB) if model_metadata['bgr'] else adversarial_image,
                                            cvtColor(image, COLOR_BGR2RGB) if model_metadata['bgr'] else image)
        dist_patches = data[0].unsqueeze(0)
        ref_patches = data[1].unsqueeze(0)
        score = self.model((dist_patches, ref_patches))
        return score.item()
