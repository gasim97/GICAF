from abc import ABCMeta, abstractmethod

class AttackInterface:
    __metaclass__ = ABCMeta

    @classmethod
    def version(cls): return "1.0"

    # initialize
    @abstractmethod
    def __init__(self): 
        raise NotImplementedError("Attack module __init__() function missing")

    # runs the attack
    @abstractmethod
    def run(self, images, model, logger, query_limit=5000): 
        raise NotImplementedError("Attack module run() function missing")
    # returns attack log

    # end of session clean up
    @abstractmethod
    def close(self): 
        pass
