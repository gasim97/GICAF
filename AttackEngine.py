from gicaf.interface.AttackEngineInterface import AttackEngineInterface
from gicaf.Logger import Logger
from numpy import array
from logging import info

# TODO: Allow memory for samples of the same class

class AttackEngine(AttackEngineInterface):

    def __init__(self, x, y, model, attacks): 
        self.x = x
        self.y = y
        self.model = model
        self.attacks = attacks
        self.loggers = []
        self._filter_wrong_predictions()

    def _filter_wrong_predictions(self):
        preds = self.model.get_top_1_batch(self.x)
        correct_pred_indicies = list(map(lambda z: z[1], filter(lambda z: z[0] == True, map(lambda i: [preds[i][0] == self.y[i], i], range(len(self.y))))))
        info(str(len(correct_pred_indicies)) + " out of " + str(len(self.x)) + " samples correctly predicted and will be used for an attack")
        self.x = array(list(map(lambda i: self.x[i], correct_pred_indicies)))
        self.y = array(list(map(lambda i: self.y[i], correct_pred_indicies)))

    # runs the attack
    def run(self): 
        for attack in self.attacks:
            self.loggers.append(Logger())
            attack.run(self.x, self.model, self.loggers[-1])
    # returns adversarial image, attack log

    def get_logs(self):
        return self.loggers

    # end of session clean up
    def close(self): 
        for logger in self.loggers:
            logger.close()
