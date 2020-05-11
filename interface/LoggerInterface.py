from abc import ABCMeta, abstractmethod

class LoggerInterface:
    __metaclass__ = ABCMeta

    @classmethod
    def version(cls): return "1.0"

    # initialize
    @abstractmethod
    def __init__(self): 
        print("Logger module __init__() function missing")
        raise NotImplementedError

    # new log
    @abstractmethod
    def nl(self, fields):
        print("Logger module nl() function missing")
        raise NotImplementedError

    # log new item
    @abstractmethod
    def append(self, data): 
        print("Logger module append() function missing")
        raise NotImplementedError

    # get current log
    @abstractmethod
    def get(self): 
        print("Logger module get() function missing")
        raise NotImplementedError
    # returns current log

    # get all logs
    @abstractmethod
    def get_all(self): 
        print("Logger module get_all() function missing")
        raise NotImplementedError
    # returns all logs

    # end of session clean up
    @abstractmethod
    def close(self):
        pass