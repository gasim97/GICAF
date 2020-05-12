from gicaf.interface.AttackEngineInterface import AttackEngineInterface
from gicaf.Logger import Logger
from numpy import array
from logging import info

class AttackEngine(AttackEngineInterface):

    # initialize
    def __init__(self, x, y, model, attacks): 
        self.x = x
        self.y = y
        self.model = model
        self.attacks = attacks
        self.logger = Logger()
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
            attack.run(self.x, self.model, self.logger)
    # returns adversarial image, attack log

    # end of session clean up
    def close(self): 
        pass
