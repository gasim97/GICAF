from Interfaces.LoggerInterface import LoggerInterface

from logging import info
from pandas import DataFrame

class Logger(LoggerInterface):

    # initialize
    def __init__(self): 
        self.logs = []

    # new log
    def nl(self, fields):
        new_log = DataFrame(columns=fields)
        self.logs.append(new_log)
        info("New log (" + str(len(self.logs)) + ") with columns:\n" + str(fields) + "\n\n")

    # log new item
    def append(self, data): 
        log = self.logs[-1]
        index = len(log)
        log.loc[index] = data
        info("Appended to log " + str(len(self.logs)) + ":\n" + str(log.loc[index]) + "\n\n")

    # get current log
    def get(self): 
        return self.logs[-1]
    # returns current log

    # get all logs
    def get_all(self): 
        return self.logs
    # returns all logs

    # end of session clean up, save all the logs
    def close(self):
        pass
