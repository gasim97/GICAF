from abc import ABCMeta, abstractmethod

class AttackEngineInterface:
    __metaclass__ = ABCMeta

    @classmethod
    def version(cls): return "1.0"

    @abstractmethod
    def __init__(self, x, y, model, attacks): 
        """
        Initialize attack engine

        Parameters
        ----------
            x : numpy.ndarray -> shape = (batch size, height, width, channels)
                Image(s) to use for attacks
            y : numpy.ndarray -> shape = (batch size, 1)
                Ground-truths of x 
            model : ModelInterface
                Model to carry out attacks on
            attacks : list with elements of type AttackInterface
                Attacks to carry out
        """
        raise NotImplementedError("AttackEngine module __init__() function missing")

    @abstractmethod
    def run(self, metric_names=None, use_memory=False): 
        """
        Runs the attack

        Parameters
        ----------
            metric_names : list with elements of type string
                The metric names of the visual metrics to be collected. Default is
                None
            use_memory : bool
                Indicates whether or not to transfer knowledge from successful
                attacks to subsequent images of the same class. Memory is not to
                be transfered between different attack methods. Default is False
        Returns
        -------
            loggers : list with elements of type LoggerInterface
                The experiment logs
        """
        raise NotImplementedError("AttackEngine module run() function missing")

    @abstractmethod
    def get_logs(self):
        """
        Get experiment logs

        Returns
        -------
            loggers : list with elements of type LoggerInterface
                The experiment logs
        """
        raise NotImplementedError("Attack module get_logs() function missing")

    @abstractmethod
    def close(self): 
        """
        End of session clean up
        """
        pass
