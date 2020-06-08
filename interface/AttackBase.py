from typing import Optional, Dict, Any, Type
from gicaf.interface.ModelBase import ModelBase
from gicaf.interface.LoggerBase import LoggerBase
from abc import ABC, abstractmethod
from numpy import ndarray

class AttackBase(ABC):

    @classmethod
    def version(cls): return "1.0"

    @abstractmethod
    def __init__(
        self, 
        attack_parameters: Optional[Dict[str, Any]] = None
    ) -> None: 
        """
        Initialize attack

        Parameters
        ----------
            attack_parameters : optional dict
                A dictionary containing attack parameters. Default is None. As
                such Attack modules must have default parameter values
        """
        ...

    @abstractmethod
    def __call__(self, 
        image: ndarray, 
        model: Type[ModelBase], 
        logger: Type[LoggerBase], 
        query_limit: int = 5000
    ) -> Optional[ndarray]: 
        """
        Runs the attack

        Parameters
        ----------
            image : numpy.ndarray
                Image to carry out attack on
            model : ModelBase
                The model to attack wrapped in an instance of ModelBase
            logger: LoggerBase
                A logger to log experimental data
            query_limit : int
                The maximum number of model queries allowable to achieve a
                successful attack. The default is 5000
        Returns
        -------
            adv : numpy.ndarray or None
                Adversarial image if successful, or None otherwise
        """
        ...
