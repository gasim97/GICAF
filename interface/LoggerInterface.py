from abc import ABCMeta, abstractmethod

class LoggerInterface:
    __metaclass__ = ABCMeta

    @classmethod
    def version(cls): return "1.0"

    @abstractmethod
    def __init__(self): 
        """
        Initialize logger
        """
        raise NotImplementedError("Logger module __init__() function missing")

    @abstractmethod
    def nl(self, fields):
        """
        Creates/begins a new log

        Parameters
        ----------
            fields : list with elements of type string
                The column headings of the new log
        """
        raise NotImplementedError("Logger module nl() function missing")

    @abstractmethod
    def append(self, data): 
        """
        Log new item

        Parameters
        ----------
            data : dict
                Data to append to the current log. The fields must match those
                provided in the last call to LoggerInterface.nl(fields)
        """
        raise NotImplementedError("Logger module append() function missing")

    @abstractmethod
    def get(self): 
        """
        Get current log

        Returns
        -------
            log : LoggerInterface
                The last experiment log created/began
        """
        raise NotImplementedError("Logger module get() function missing")

    @abstractmethod
    def get_all(self):
        """
        Get all logs

        Returns
        -------
            logs : list with elements of type LoggerInterface
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
