from typing import Optional, Type, List, Mapping, Any
from gicaf.interface.LoggerBase import LoggerBase
from gicaf.interface.MetricCollectorBase import MetricCollectorBase
from numpy import ndarray
from pandas import DataFrame
from logging import info
from pickle import dump, load
from pathlib import Path, PosixPath
from os.path import dirname

class Logger(LoggerBase):
    
    def __init__(
        self, 
        metric_collector: Optional[Type[MetricCollectorBase]] = None
    ) -> None: 
        self.logs = []
        self.saved = False
        self.metric_collector = metric_collector

    # new log
    def nl(
        self, 
        fields: List[str]
    ) -> None:
        if self.metric_collector:
            metric_list = self.metric_collector.get_metric_list()
            for metric in metric_list:
                fields.append(metric)
        new_log = DataFrame(columns=fields)
        self.logs.append(new_log)
        info("New log (" + str(len(self.logs)) + ") with columns:\n" + str(fields) + "\n\n")

    # log new item
    def append(
        self, 
        data: Mapping[str, Any], 
        image: ndarray, 
        adversarial_image: ndarray
    ) -> None: 
        self.saved = False
        metric_results = self.metric_collector(image, adversarial_image)
        data.update(metric_results)
        log = self.logs[-1]
        index = len(log)
        log.loc[index] = data
        info("Appended to log " + str(len(self.logs)) + ":\n" + str(log.loc[index]) + "\n\n")

    # get current log
    def get(self) -> DataFrame: 
        return self.logs[-1]

    # get all logs
    def get_all(self) -> List[DataFrame]:
        return self.logs

    def _save_dir(self) -> PosixPath:
        save_dir = Path(dirname(__file__) + "/tmp/results/")
        save_dir.mkdir(exist_ok=True, parents=True)
        return save_dir

    def _save_file(self) -> PosixPath:
        save_dir = self._save_dir()
        save_dir_str = str(save_dir)
        files = [str(file_) for file_ in save_dir.iterdir()]
        experiments = list(map(
            lambda exp: int(exp.split('-')[1].split('.')[0]), 
            filter(lambda f: f.split('-')[0] == (save_dir_str + "/experiment"), files)
        ))
        experiment_id = 1
        if (len(experiments) > 0):
            experiment_id += max(experiments)
        self.save_file = save_dir/("experiment-" + str(experiment_id) + ".txt")
        return self.save_file

    def save(self) -> None:
        if (self.saved):
            info("Experiment logs already saved to " + str(self.save_file))
            return
        save_file = str(self._save_file())
        with open(save_file, "wb") as fn: 
            dump(self.logs, fn)
        info("Experiment logs saved to " + save_file)
        self.saved = True

    def load(
        self, 
        experiment_id: int
    ) -> None:
        load_file = str(self._save_dir()/("experiment-" + str(experiment_id) + ".txt"))
        with open(load_file, "rb") as fn: 
            self.logs = load(fn)
        self.save_file = load_file
        self.saved = True
        info("Experiment logs loaded from " + load_file + "\nRun 'logger.get_all()' to get the loaded logs")

    # end of session clean up, save all the logs
    def close(self) -> None:
        self.save()
