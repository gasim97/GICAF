from typing import List, Optional, Type
from gicaf.interface.ModelBase import ModelBase
from abc import ABC, abstractmethod

class MetricCollectorBase(ABC):

    @classmethod
    def version(cls): return "1.0"

    @abstractmethod
    def __init__(
        self, 
        model: Type[ModelBase], 
        metric_names: Optional[List[str]] = None
    ) -> None: 
        """
        Initialize visual quality assessment metrics collector

        Parameters
        ----------
            model : ModelBase
                The model instance to be used in the attack
            metric_names : list with elements of type string
                The names of the metrics to be collected from the internal list that
                maps metric names to metric classes. Default is None
        Raises
        ------
            NameError if an invalid metric name is provided
        Note
        ----
            This method must create the following variables:
                self.metadata = metadata
        """
        ...

    @abstractmethod
    def __call__(self, image, adversarial_image): 
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
        Note
        ----
            This method must also add the current model query count as to the result dictionary, 
            e.g.: 
                'model queries': self.model.get_query_count()
        """
        ...

    @abstractmethod
    def get_metric_list(self):
        """
        Get the list of metrics to be collected

        Returns
        -------
            metric_names : list with elements of type string
                The metric names of the metrics to be collected
        """
        ...
