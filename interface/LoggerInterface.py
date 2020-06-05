from abc import ABCMeta, abstractmethod

class LoggerInterface:
    __metaclass__ = ABCMeta

    @classmethod
    def version(cls): return "1.0"

    @abstractmethod
    def __init__(self, metric_collector): 
        """
        Initialize logger

        Parameters
        ----------
            metric_collector : MetricCollector
                Initialised metric collector, which should be used to colle
        """
        raise NotImplementedError("Logger module __init__() function missing")

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
        raise NotImplementedError("Logger module nl() function missing")

    @abstractmethod
    def append(self, data, image, adversarial_image): 
        """
        Run metrics and log new item

        Parameters
        ----------
            data : dict
                Data to append to the current log. The fields must match those
                provided in the last call to LoggerInterface.nl(fields)
            image : numpy.ndarray
                Reference image
            adversarial_image : numpy.ndarray
                Adversarial image at current step
        """
        raise NotImplementedError("Logger module append() function missing")

    @abstractmethod
    def get(self): 
        """
        Get current log

        Returns
        -------
            log : pandas.DataFrame
                The last experiment log created/began
        """
        raise NotImplementedError("Logger module get() function missing")

    @abstractmethod
    def get_all(self):
        """
        Get all logs

        Returns
        -------
            logs : list with elements of type pandas.DataFrame
                The experiment logs
        """ 
        raise NotImplementedError("Logger module get_all() function missing")

    @abstractmethod
    def save(self):
        """
        Saves the experiment logs
        """
        raise NotImplementedError("Logger module save() function missing")

    @abstractmethod
    def load(self, id):
        """
        Loads saved experiment logs

        Parameters
        ----------
            id : Any
                An identifier indicating which experiment logs to load
        """
        raise NotImplementedError("Logger module load() function missing")

    @abstractmethod
    def close(self):
        """
        End of session clean up
        """
        pass
