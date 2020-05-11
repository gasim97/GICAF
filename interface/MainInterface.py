from abc import ABCMeta, abstractmethod

class MainInterface:
    __metaclass__ = ABCMeta

    @classmethod
    def version(cls): return "1.0"

    # initialize
    @abstractmethod
    def __init__(self): 
        print("Main module __init__() function missing")
        raise NotImplementedError

    # end of session clean up
    @abstractmethod
    def close(self): 
        print("Main module close() function missing")
        raise NotImplementedError
