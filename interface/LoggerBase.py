from abc import ABC, abstractmethod

class LoggerBase(ABC):

    @classmethod
    def version(cls): return "1.0"

    @abstractmethod
    def __init__(self, metric_collector=None): 
        """
        Initialize logger

        Parameters
        ----------
            metric_collector : MetricCollector
                Initialised metric collector. Default is None
        """
        ...

    @abstractmethod
    def nl(self, fields):
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
    def append(self, data, image, adversarial_image): 
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
    def get(self): 
        """
        Get current log

        Returns
        -------
            log : pandas.DataFrame
                The last experiment log created/began
        """
        ...

    @abstractmethod
    def get_all(self):
        """
        Get all logs

        Returns
        -------
            logs : list with elements of type pandas.DataFrame
                The experiment logs
        """ 
        ...

    @abstractmethod
    def save(self):
        """
        Saves the experiment logs
        """
        ...

    @abstractmethod
    def load(self, id):
        """
        Loads saved experiment logs

        Parameters
        ----------
            id : Any
                An identifier indicating which experiment logs to load
        """
        ...

    @abstractmethod
    def close(self):
        """
        End of session clean up
        """
        ...
