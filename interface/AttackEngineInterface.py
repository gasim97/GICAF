from typing import Callable, List, Tuple, Optional
from gicaf.interface.ModelInterface import ModelInterface
from gicaf.interface.AttackInterface import AttackInterface
from gicaf.interface.LoggerInterface import LoggerInterface
import numpy as np
from abc import ABCMeta, abstractmethod

class AttackEngineInterface:
    __metaclass__ = ABCMeta

    @classmethod
    def version(cls): return "1.0"

    @abstractmethod
    def __init__(
        self, 
        data_generator: Callable[None, Tuple[np.ndarray, int]], 
        model: ModelInterface, 
        attacks: List[AttackInterface],
        save: bool = True
    ) -> None: 
        """
        Initialize attack engine

        Parameters
        ----------
            data_generator : generator function
                Provides the samples loaded by the user
                Yields
                ------
                    x : numpy.ndarray
                        Image to use for attacks
                    y : int
                        Ground-truth of x 
            model : ModelInterface
                Model to carry out attacks on
            attacks : list with elements of type AttackInterface
                Attacks to carry out
        """
        raise NotImplementedError("AttackEngine module __init__() function missing")

    @abstractmethod
    def run(
        self, 
        metric_names: Optional[List[str]] = None, 
        use_memory: bool = False
    ) -> Tuple[List[LoggerInterface], List[float]]: 
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
            success_rates : list with elements of type float
                The experiment adversarial success rates in percentage
        Note
        ----
            This method must call 'self.model.reset_query_count()' before each attack to
            reset the model's query count
        """
        raise NotImplementedError("AttackEngine module run() function missing")

    @abstractmethod
    def get_logs(self) -> List[LoggerInterface]:
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
