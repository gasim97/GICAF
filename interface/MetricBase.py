from typing import Mapping, Union, Tuple
from abc import ABC, abstractmethod
from numpy import ndarray
# Note: subclasses must be added to the metric_list in MetricCollector.py
class MetricBase(ABC):

    @classmethod
    def version(cls) -> str: return "1.0"

    def __init__(self) -> None:
        """
        Initialize metric
        """
        pass

    @abstractmethod
    def __call__(
        self, 
        image: ndarray, 
        adversarial_image: ndarray, 
        model_metadata: Mapping[str, Union[int, bool, Tuple[int, int]]]
    ) -> float: 
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
