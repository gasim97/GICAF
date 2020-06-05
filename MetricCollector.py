from gicaf.interface.MetricCollectorInterface import MetricCollectorInterface
import gicaf.metrics as metrics

metric_list = {
    "absolute-value norm": metrics.pnorm.PNorm.AbsValueNorm,
    "euclidean norm": metrics.pnorm.PNorm.EuclideanNorm,
    "infinity norm": metrics.pnorm.PNorm.InfNorm,
    "psnr": metrics.psnr.PSNR.PSNR,
    "ssim": metrics.ssim.SSIM.SSIM,
    "wadiqam": metrics.wadiqam.WaDIQaM.WaDIQaM,
}

class MetricCollector(MetricCollectorInterface):

    @classmethod
    def supported_metrics(cls): return metric_list.keys

    def __init__(self, model_metadata, metric_names=None):
        if metric_names == None:
            self.metrics = []
            return
        try: 
            self.metrics = list(map(lambda x: (x, metric_list[x]), metric_names))
        except KeyError:
            raise NameError("Invalid metric name provided.\n The metrics below are supported:\n" + str(metric_list.keys))
        self.metric_names = metric_names
        self.model_metadata = model_metadata

    def get_metric_list(self):
        if self.metrics == []:
            return
        return self.metric_names

    def collect_metrics(self, image, adversarial_image): 
        result = {}
        for metric in self.metrics:
            result[metric[0]] = metric[1](image, adversarial_image, self.model_metadata)
        return result
