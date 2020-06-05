from gicaf.interface.MetricCollectorInterface import MetricCollectorInterface
import gicaf.metrics as metrics

metric_list = {
    "absolute-value norm": metrics.PNorm.AbsValueNorm(),
    "euclidean norm": metrics.PNorm.EuclideanNorm(),
    "infinity norm": metrics.PNorm.InfNorm(),
    "psnr": metrics.PSNR.PSNR(),
    "ssim": metrics.SSIM.SSIM(),
    "wadiqam": metrics.WaDIQaM.WaDIQaM(),
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
