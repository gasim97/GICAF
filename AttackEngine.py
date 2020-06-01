from gicaf.interface.AttackEngineInterface import AttackEngineInterface
from gicaf.Logger import Logger
from numpy import array
from copy import deepcopy
from logging import info

class AttackEngine(AttackEngineInterface):

    def __init__(self, x, y, model, attacks, save=True): 
        self.x = x
        self.y = y
        self.model = model
        self.attacks = attacks
        self.loggers = []
        self.success_rates = []
        self.memory = {}
        self.save = save
        self._filter_wrong_predictions()

    def _filter_wrong_predictions(self):
        preds = self.model.get_top_1_batch(self.x)
        correct_pred_indicies = list(map(lambda z: z[1], filter(lambda z: z[0] == True, map(lambda i: [preds[i][0] == self.y[i], i], range(len(self.y))))))
        info(str(len(correct_pred_indicies)) + " out of " + str(len(self.x)) + " samples correctly predicted and will be used for an attack")
        self.x = array(list(map(lambda i: self.x[i], correct_pred_indicies)))
        self.y = array(list(map(lambda i: self.y[i], correct_pred_indicies)))

    def run(self, use_memory=False): 
        for attack in self.attacks:
            self.loggers.append(Logger())
            self.memory = {}
            for i, image in enumerate(deepcopy(self.x)):
                if use_memory:
                    try:
                        image = image + self.memory[str(self.y[i])]
                    except KeyError:
                        pass
                adv = attack.run(image, self.model, self.loggers[-1])
                if use_memory:
                    self.memory[str(self.y[i])] = adv - image
        return self.loggers

    def get_logs(self):
        return self.loggers

    def close(self):
        if self.save:
            for attack in self.attacks:
                attack.close() 
            for logger in self.loggers:
                logger.close()
