from abc import ABCMeta, abstractmethod

class LoggerInterface:
    __metaclass__ = ABCMeta

    @classmethod
    def version(cls): return "1.0"

    # initialize
    @abstractmethod
    def __init__(self): 
        raise NotImplementedError("Logger module __init__() function missing")

    # new log
    @abstractmethod
    def nl(self, fields):
        raise NotImplementedError("Logger module nl() function missing")

    # log new item
    @abstractmethod
    def append(self, data): 
        raise NotImplementedError("Logger module append() function missing")

    # get current log
    @abstractmethod
    def get(self): 
        raise NotImplementedError("Logger module get() function missing")
    # returns current log

    # get all logs
    @abstractmethod
    def get_all(self): 
        raise NotImplementedError("Logger module get_all() function missing")
    # returns all logs

    @abstractmethod
    def save(self):
        raise NotImplementedError("Logger module save() function missing")

    @abstractmethod
    def load(self, experiment_id):
        raise NotImplementedError("Logger module load() function missing")

    # end of session clean up
    @abstractmethod
    def close(self):
        pass
