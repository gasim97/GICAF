from typing import Type, Optional, List, Union, Dict
from gicaf.interface.MetricCollectorBase import MetricCollectorBase
from gicaf.interface.ModelBase import ModelBase
from numpy import ndarray
import gicaf.metrics.PNorm as PNorm
import gicaf.metrics.PSNR as PSNR
import gicaf.metrics.SSIM as SSIM
import gicaf.metrics.WaDIQaM as WaDIQaM

metric_list = {
    "absolute-value norm": PNorm.AbsValueNorm,
    "euclidean norm": PNorm.EuclideanNorm,
    "infinity norm": PNorm.InfNorm,
    "psnr": PSNR.PSNR,
    "ssim": SSIM.SSIM,
    "wadiqam": WaDIQaM.WaDIQaM,
}

class MetricCollector(MetricCollectorBase):

    @classmethod
    def supported_metrics(cls): return metric_list.keys()

    def __init__(
        self, 
        model: Type[ModelBase], 
        metric_names: Optional[List[str]] = None
    ) -> None: 
        self.metric_names = ['model queries']
        if metric_names == None:
            self.metrics = []
            return
        try: 
            self.metrics = list(map(lambda x: (x, metric_list[x]()), metric_names))
        except KeyError as e:
            raise NameError("Invalid metric name provided - " + str(e) + ".\n The metrics below are supported:\n" 
                            + str(metric_list.keys()))
        for name in metric_names:
          self.metric_names.append(name)
        self.model = model

    def __call__(
        self, 
        image: ndarray, 
        adversarial_image: ndarray
    ) -> Dict[str, Union[float, int]]: 
        result = {
            'model queries': self.model.get_query_count()
        }
        for metric in self.metrics:
            result[metric[0]] = metric[1](image, adversarial_image, self.model.metadata)
        return result

    def get_metric_list(self) -> List[str]:
        return self.metric_names
