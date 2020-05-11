from abc import ABCMeta, abstractmethod

class AttackInterface:
    __metaclass__ = ABCMeta

    @classmethod
    def version(cls): return "1.0"

    # initialize
    @abstractmethod
    def __init__(self): 
        print("Attack module __init__() function missing")
        raise NotImplementedError

    # runs the attack
    @abstractmethod
    def run(self, images, model, logger, query_limit=5000): 
        print("Attack module run() function missing")
        raise NotImplementedError
    # returns attack log

    # end of session clean up
    @abstractmethod
    def close(self): 
        pass
