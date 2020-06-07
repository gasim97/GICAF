from typing import Callable, List, Tuple, Optional
from gicaf.interface.AttackEngineInterface import AttackEngineInterface
from gicaf.interface.ModelInterface import ModelInterface
from gicaf.interface.AttackInterface import AttackInterface
from gicaf.interface.LoggerInterface import LoggerInterface
from gicaf.Logger import Logger
from gicaf.MetricCollector import MetricCollector
import numpy as np
from copy import deepcopy
from logging import info

class AttackEngine(AttackEngineInterface):

    def __init__(
        self, 
        data_generator: Callable[None, Tuple[np.ndarray, int]], 
        model: ModelInterface, 
        attacks: List[AttackInterface],
        save: bool = True
    ) -> None:  
        self.data_generator = data_generator
        self.model = model
        self.attacks = attacks
        self.loggers = []
        self.success_rates = []
        self.save = save
        self.pred_result_indicies = {
            'correct': [],
            'incorrect': [],
        }
        self._filter_predictions()

    def _filter_predictions(self):
        for i, (x, y) in enumerate(self.data_generator()):
            if self.model.get_top_1(x)[0] == y:
                self.pred_result_indicies['correct'].append(i)
            else:
                self.pred_result_indicies['incorrect'].append(i)
            count = i
        info(str(len(self.pred_result_indicies['correct'])) + " out of " + str(count + 1) + 
            " samples correctly predicted and will be used for an attack")

    def run(
        self, 
        metric_names: Optional[List[str]] = None, 
        use_memory: bool = False
    ) -> Tuple[List[LoggerInterface], List[float]]: 
        metric_collector = MetricCollector(self.model, metric_names)
        for attack in self.attacks:
            self.loggers.append(Logger(metric_collector=metric_collector))
            memory = {}
            num_success = 0
            for i, (x, y) in enumerate(self.data_generator()):
                if i in self.pred_result_indicies['correct']:
                    if use_memory:
                        try:
                            x = x + memory[str(y)]
                        except KeyError:
                            pass
                    self.model.reset_query_count()
                    adv = attack(x, self.model, self.loggers[-1])
                    if type(adv) == type(None):
                        num_success += 1
                        if use_memory:
                            memory[str(y)] = adv - x
            self.success_rates.append(100*num_success/len(self.pred_result_indicies['correct']))
        return self.loggers, self.success_rates

    def get_logs(self):
        return self.loggers

    def close(self):
        if self.save:
            for attack in self.attacks:
                attack.close() 
            for logger in self.loggers:
                logger.close()
