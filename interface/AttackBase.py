from typing import Optional, Dict, Any, Type
from gicaf.interface.ModelBase import ModelBase
from gicaf.interface.LoggerBase import LoggerBase
from abc import ABC, abstractmethod
from numpy import ndarray

class AttackBase(ABC):

    @classmethod
    def version(cls) -> str: return "1.0"

    @abstractmethod
    def __init__(
        self, 
        **kwargs
    ) -> None: 
        """
        Initialize attack

        Parameters
        ----------
            Named parameters are to be defined by sub-classes, where needed. Furthermore,
            sub-classes should have and use default paramters where the user does not specify
            a parameter
        """
        ...

    @abstractmethod
    def __call__(self, 
        image: ndarray, 
        model: Type[ModelBase], 
        logger: Type[LoggerBase], 
        ground_truth: Optional[int] = None,
        target: Optional[int] = None,
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
            ground_truth : optional int
                The original class, if None then a false positive attack is carried
                out
            target : optional int
                The targeted class, if None then an untargeted attack is carried
                out
            query_limit : int
                The maximum number of model queries allowable to achieve a
                successful attack. The default is 5000
        Returns
        -------
            adv : numpy.ndarray or None
                Adversarial image if successful, or None otherwise
        Raises
        ------
            NotImplementedError
                When the attack setting infered from the optional arguments is
                supported by the algorithm, but has not yet been implemented
            ValueError
                When the attack setting infered from the optional arguments is not
                supported by the algorithm
        """
        ...
