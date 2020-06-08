from typing import Mapping, Union, Tuple
from gicaf.interface.MetricBase import MetricBase
from gicaf.metrics.resources.wadiqam import Utils
from numpy import ndarray
from cv2 import cvtColor, COLOR_BGR2RGB

class WaDIQaM(MetricBase):

    def __init__(self) -> None:
        self.model = Utils.get_FRnet()

    def __call__(
        self, 
        image: ndarray, 
        adversarial_image: ndarray, 
        model_metadata: Mapping[str, Union[int, bool, Tuple[int, int]]]
    ) -> float: 
        data = Utils.NonOverlappingCropPatches(cvtColor(adversarial_image, COLOR_BGR2RGB) if model_metadata['bgr'] else adversarial_image,
                                            cvtColor(image, COLOR_BGR2RGB) if model_metadata['bgr'] else image)
        dist_patches = data[0].unsqueeze(0)
        ref_patches = data[1].unsqueeze(0)
        score = self.model((dist_patches, ref_patches))
        return score.item()
