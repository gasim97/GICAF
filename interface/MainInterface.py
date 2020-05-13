from abc import ABCMeta, abstractmethod

class MainInterface:
    __metaclass__ = ABCMeta

    @classmethod
    def version(cls): return "1.0"

    # initialize
    @abstractmethod
    def __init__(self): 
        raise NotImplementedError("Main module __init__() function missing")

    # end of session clean up
    @abstractmethod
    def close(self): 
        raise NotImplementedError("Main module close() function missing")
