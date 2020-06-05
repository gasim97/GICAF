from abc import ABCMeta, abstractmethod

class MetricCollectorInterface:
    __metaclass__ = ABCMeta

    @classmethod
    def version(cls): return "1.0"

    @abstractmethod
    def __init__(self, model_metadata, metric_names=None): 
        """
        Initialize visual quality assessment metrics collector

        Parameters
        ----------
            model_metadata : dict
                Model metadata dictionary populated as specified in ModelInterface.py
            metric_names : list with elements of type string
                The names of the metrics to be collected from the internal list that
                maps metric names to metric classes. Default is None
        Raises
        ------
            NameError if an invalid metric name is provided
        """
        raise NotImplementedError("MetricCollector module __init__() function missing")

    @abstractmethod
    def get_metric_list(self):
        """
        Get the list of metrics to be collected

        Returns
        -------
            metric_names : list with elements of type string
                The metric names of the metrics to be collected
        """
        raise NotImplementedError("MetricCollector module get_metrics() function missing")

    @abstractmethod
    def collect_metrics(self, image, adversarial_image): 
        """
        Collect metrics on samples

        Parameters
        ----------
            image : numpy.ndarray
                Reference image
            adversarial_image : numpy.ndarray
                Adversarial image at current step
        Returns
        -------
            result : dict
                Dictionary with metric names as keys and their evaluation on the input data
                as values
        """
        raise NotImplementedError("MetricCollector module collect_metrics() function missing")
