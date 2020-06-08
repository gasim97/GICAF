from typing import Optional, Type, List, Any, Mapping
from gicaf.interface.MetricCollectorBase import MetricCollectorBase
from abc import ABC, abstractmethod
from numpy import ndarray
from pandas import DataFrame

class LoggerBase(ABC):

    @classmethod
    def version(cls): return "1.0"

    @abstractmethod
    def __init__(
        self, 
        metric_collector: Optional[Type[MetricCollectorBase]] = None
    ) -> None: 
        """
        Initialize logger

        Parameters
        ----------
            metric_collector : MetricCollector
                Initialised metric collector. Default is None
        """
        ...

    @abstractmethod
    def nl(
        self, 
        fields: List[str]
    ) -> None:
        """
        Creates/begins a new log

        Parameters
        ----------
            fields : list with elements of type string
                The fields of the new log. The metric names of the metrics to
                be collected by the MetricCollector should be also be fields
        """
        ...

    @abstractmethod
    def append(
        self, 
        data: Mapping[str, Any], 
        image: ndarray, 
        adversarial_image: ndarray
    ) -> None: 
        """
        Run metrics and log new item

        Parameters
        ----------
            data : dict
                Data to append to the current log. The fields must match those
                provided in the last call to LoggerBase.nl(fields)
            image : numpy.ndarray
                Reference image
            adversarial_image : numpy.ndarray
                Adversarial image at current step
        """
        ...

    @abstractmethod
    def get(self) -> DataFrame: 
        """
        Get current log

        Returns
        -------
            log : pandas.DataFrame
                The last experiment log created/began
        """
        ...

    @abstractmethod
    def get_all(self) -> List[DataFrame]:
        """
        Get all logs

        Returns
        -------
            logs : list with elements of type pandas.DataFrame
                The experiment logs
        """ 
        ...

    @abstractmethod
    def save(self) -> None:
        """
        Saves the experiment logs
        """
        ...

    @abstractmethod
    def load(
        self, 
        id: Any
    ) -> None:
        """
        Loads saved experiment logs

        Parameters
        ----------
            id : Any
                An identifier indicating which experiment logs to load
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """
        End of session clean up
        """
        ...
