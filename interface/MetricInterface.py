from abc import ABCMeta, abstractmethod

class MetricInterface:
    __metaclass__ = ABCMeta

    @classmethod
    def version(cls): return "1.0"

    @abstractmethod
    def __init__(self):
        """
        Initialize metric
        """
        pass

    @abstractmethod
    def __call__(self, image, adversarial_image, model_metadata): 
        """
        Run visual quality assessment metric

        Parameters
        ----------
            image : numpy.ndarray
                Reference image
            adversarial_image : numpy.ndarray
                Adversarial image at current step
            model_metadata : dict
                Model metadata dictionary populated as specified in ModelInterface.py
        Returns
        -------
            result : float
                Result
        """
        raise NotImplementedError("Metric module __call__() function missing")