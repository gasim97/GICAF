from gicaf.interface.LoggerInterface import LoggerInterface
from pandas import DataFrame
from logging import info
from pickle import dump, load
from pathlib import Path
from os.path import dirname

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

    def _save_dir(self):
        save_dir = Path(dirname(__file__) + "/tmp/results/")
        save_dir.mkdir(exist_ok=True, parents=True)
        return save_dir

    def _save_file(self):
        save_dir = self._save_dir()
        experiements = [int(str(exp).split('-')[1]) for exp in save_dir.iterdir()]
        experiment_id = 1
        if (len(experiements) > 0):
            experiment_id += max(experiements)
        return save_dir/("experiement-" + str(experiment_id) + ".txt")

    def save(self):
        save_file = str(self._save_file())
        with open(save_file, "wb") as fn: 
            dump(self.logs, fn)
        info("Experiment logs saved to " + save_file)

    def load(self, experiement_id):
        load_file = str(self._save_dir()/("experiement-" + str(experiement_id) + ".txt"))
        with open(load_file, "wb") as fn: 
            self.logs = load(fn)
        info("Experiment logs loaded from " + load_file + "\nRun 'logger.get_all()' to get the loaded logs")

    # end of session clean up, save all the logs
    def close(self):
        self.save()
