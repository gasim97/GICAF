from GICAF.Interfaces.AttackEngineInterface import AttackEngineInterface
from Logger import Logger

class AttackEngine(AttackEngineInterface):

    # initialize
    def __init__(self, x, y, model, attacks): 
        self.x = x
        self.y = y
        self.model = model
        self.attacks = attacks
        self.logger = Logger()

    # runs the attack
    def run(self): 
        for attack in self.attacks:
            attack.run(self.x, self.model, self.logger)
    # returns adversarial image, attack log

    # end of session clean up
    def close(self): 
        pass
