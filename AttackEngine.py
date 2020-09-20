from typing import Callable, List, Tuple, Optional, Type
from gicaf.interface.AttackEngineBase import AttackEngineBase
from gicaf.interface.ModelBase import ModelBase
from gicaf.interface.AttackBase import AttackBase
from gicaf.interface.LoggerBase import LoggerBase
from gicaf.Logger import Logger
from gicaf.MetricCollector import MetricCollector
from numpy import ndarray
from copy import deepcopy
from logging import info, warning

class AttackEngine(AttackEngineBase):

    def __init__(
        self, 
        data_generator: Callable[[None], Tuple[ndarray, int]], 
        model: Type[ModelBase], 
        attacks: List[Type[AttackBase]],
        targets: Optional[List[int]] = None,
        false_positive: bool = True,
        save: bool = True
    ) -> None:
        self.data_generator = data_generator
        self.model = model
        self.attacks = attacks
        self.targets = targets
        self.loggers = []
        self.success_rates = []
        self.false_positive = false_positive
        self.save = save
        self.closed = False
        self.pred_result_indicies = {
            'correct': [],
            'incorrect': [],
        }
        if false_positive:
            self._filter_predictions()

    def _filter_predictions(self) -> None:
        for i, (x, y) in enumerate(self.data_generator()):
            if self.model.get_top_1(x)[0] == y:
                self.pred_result_indicies['correct'].append(i)
            count = i
        info(str(len(self.pred_result_indicies['correct'])) + " out of " + str(count + 1) + 
            " samples correctly predicted and will be used for an attack")

    def run(
        self, 
        metric_names: Optional[List[str]] = None, 
        use_memory: bool = False,
        query_limit: int = 5000
    ) -> Tuple[List[Type[LoggerBase]], List[float]]: 
        if self.closed:
            info("Cannot run attack engine after it has been closed")
            return self.loggers, self.success_rates
        metric_collector = MetricCollector(self.model, metric_names)
        for attack in self.attacks:
            self.loggers.append(Logger(metric_collector=metric_collector))
            memory = {}
            num_success = 0
            for i, (x, y) in enumerate(self.data_generator()):
                if not self.false_positive or i in self.pred_result_indicies['correct']:
                    if use_memory and str(y) in memory:
                        x = x + memory[str(y)]
                    self.model.reset_query_count()
                    adv = attack(
                        image=x, 
                        model=self.model, 
                        logger=self.loggers[-1], 
                        ground_truth=y,
                        target=self.targets if type(self.targets) == type(None) else self.targets[i],
                        query_limit=query_limit
                    )
                    if type(adv) != type(None):
                        num_success += 1
                        if use_memory:
                            memory[str(y)] = adv - x
            self.success_rates.append(100*num_success/len(self.pred_result_indicies['correct']))
        return self.loggers, self.success_rates

    def get_logs(self) -> List[Type[LoggerBase]]:
        return self.loggers

    def close(self) -> None:
        if self.save and not self.closed:
            for logger in self.loggers:
                logger.close()
        self.closed = True
