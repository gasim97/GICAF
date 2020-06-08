from abc import ABC, abstractmethod

class MetricBase(ABC):

    @classmethod
    def version(cls): return "1.0"

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
                Model metadata dictionary populated as specified in ModelBase.py
        Returns
        -------
            result : float
                Result
        """
        ...
