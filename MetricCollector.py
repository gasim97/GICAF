from gicaf.interface.MetricCollectorInterface import MetricCollectorInterface
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

class MetricCollector(MetricCollectorInterface):

    @classmethod
    def supported_metrics(cls): return metric_list.keys()

    def __init__(self, model, metric_names=None):
        if metric_names == None:
            self.metrics = []
            return
        try: 
            self.metrics = list(map(lambda x: (x, metric_list[x]()), metric_names))
        except KeyError as e:
            raise NameError("Invalid metric name provided - " + str(e) + ".\n The metrics below are supported:\n" 
                            + str(metric_list.keys()))
        self.metric_names = metric_names
        self.model = model

    def __call__(self, image, adversarial_image): 
        result = {
            'model queries': self.model.get_query_count()
        }
        for metric in self.metrics:
            result[metric[0]] = metric[1](image, adversarial_image, self.model.metadata)
        return result

    def get_metric_list(self):
        if self.metrics == []:
            return
        return self.metric_names
