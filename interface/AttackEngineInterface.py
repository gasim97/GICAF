from abc import ABCMeta, abstractmethod

class AttackEngineInterface:
    __metaclass__ = ABCMeta

    @classmethod
    def version(cls): return "1.0"

    # initialize
    @abstractmethod
    def __init__(self, x, y, model, attacks): 
        print("AttackEngine module __init__() function missing")
        raise NotImplementedError

    # runs the attack
    @abstractmethod
    def run(self): 
        print("AttackEngine module run() function missing")
        raise NotImplementedError
    # returns adversarial image, attack log

    # end of session clean up
    @abstractmethod
    def close(self): 
        pass
