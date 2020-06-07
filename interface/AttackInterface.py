from typing import Optional, Dict, Any
from gicaf.interface.ModelInterface import ModelInterface
from gicaf.interface.LoggerInterface import LoggerInterface
import numpy as np
from abc import ABCMeta, abstractmethod

class AttackInterface:
    __metaclass__ = ABCMeta

    @classmethod
    def version(cls): return "1.0"

    @abstractmethod
    def __init__(self, attack_parameters: Optional[Dict[str, Any]] = None) -> None: 
        """
        Initialize attack

        Parameters
        ----------
            attack_parameters : optional dict
                A dictionary containing attack parameters. Default is None. As
                such Attack modules must have default parameter values
        """
        raise NotImplementedError("Attack module __init__() function missing")

    @abstractmethod
    def __call__(self, 
        image: np.ndarray, 
        model: ModelInterface, 
        logger: LoggerInterface, 
        query_limit: int = 5000
    ) -> Optional[np.ndarray]: 
        """
        Runs the attack

        Parameters
        ----------
            image : numpy.ndarray
                Image to carry out attack on
            model : ModelInterface
                The model to attack wrapped in an instance of ModelInterface
            logger: LoggerInterface
                A logger to log experimental data
            query_limit : int
                The maximum number of model queries allowable to achieve a
                successful attack. The default is 5000
        Returns
        -------
            adv : numpy.ndarray or None
                Adversarial image if successful, or None otherwise
        """
        raise NotImplementedError("Attack module run() function missing")

    @abstractmethod
    def close(self): 
        """
        End of session clean up
        """
        pass
