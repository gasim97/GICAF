from abc import ABCMeta, abstractmethod

class AttackEngineInterface:
    __metaclass__ = ABCMeta

    @classmethod
    def version(cls): return "1.0"

    # initialize
    @abstractmethod
    def __init__(self, x, y, model, attacks): 
        raise NotImplementedError("AttackEngine module __init__() function missing")

    # runs the attack
    @abstractmethod
    def run(self, use_memory=False): 
        raise NotImplementedError("AttackEngine module run() function missing")
    # returns adversarial image, attack log

    # end of session clean up
    @abstractmethod
    def close(self): 
        pass
