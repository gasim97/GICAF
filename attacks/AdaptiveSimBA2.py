from typing import Optional, Type
from gicaf.interface.AttackBase import AttackBase
from gicaf.interface.ModelBase import ModelBase
from gicaf.interface.LoggerBase import LoggerBase
from numpy import clip, argwhere, zeros, array, ndarray
from sys import setrecursionlimit
from numpy.linalg import norm
from numpy.random import randint, uniform
import time

class AdaptiveSimBA2(AttackBase):

    def __init__(
        self, 
        size: int = 1, 
        epsilon: int = 64, 
        epsilon_multiplier: int = 2
    ) -> None: 
        self.size = size
        self.epsilon = epsilon*epsilon_multiplier
        self.initial_epsilon = epsilon
        self.epsilon_multiplier = epsilon_multiplier

    def __call__(self, 
        image: ndarray, 
        model: Type[ModelBase], 
        logger: Type[LoggerBase], 
        ground_truth: Optional[int] = None,
        target: Optional[int] = None,
        query_limit: int = 5000
    ) -> Optional[ndarray]: 
        if target:
            raise NotImplementedError("Targeted Adaptive SimBA has not been implemented yet") 
        if ground_truth == None:
            raise ValueError('Adaptive SimBA is not intended for generating false positives, please provide a ground truth')
        self.model = model
        self.height = self.model.metadata['height']
        self.width = self.model.metadata['width']
        self.channels = self.model.metadata['channels']
        self.bounds = self.model.metadata['bounds']
        self.logger = logger
        self.query_limit = query_limit
        loss_label = ground_truth

        setrecursionlimit(max(1000, int(self.height*self.width*self.channels/self.size/self.size*10))) #for deep recursion diretion sampling

        top_preds = self.model.get_top_5(image)
        top_1_label, p = top_preds[0]

        self.logger.nl(['iterations', 'epsilon','size', 
                        'is_adv', 'image', 'top_preds', 'success'])

        self.ps = [p]
        count = 0
        past_qs = []
        self.done = []
        self.num_directions = 1
        self.total_calls = 0
        delta = 0
        is_adv = self.is_adversarial(top_1_label, loss_label)
        iteration = 0
        done = []
        
        # log step 0
        adv = clip(image + delta, self.bounds[0], self.bounds[1])

        self.logger.append({
            "iterations": iteration,
            "epsilon": self.epsilon,
            "size": self.size,
            "is_adv": is_adv,
            "image": image,
            "top_preds": top_preds,
            "success": False,
        }, image, adv)

        while ((not is_adv) & (self.model.get_query_count() <= self.query_limit)): 
            iteration += 1    

            q, done = self.new_q_direction(done)

            delta, p, top_preds, success = self.check_pos(image, delta, q, p, loss_label)
            if success:
                q = -q
            self.total_calls += 1
            count +=1
            if not success:
                delta, p, top_preds, success = self.check_neg(image, delta, q, p, loss_label)
                self.total_calls += 1
                count +=1

            adv = clip(image + delta, self.bounds[0], self.bounds[1])

            if self.model.get_query_count() % 250 < 2 and self.model.get_query_count < 1002:
                self.epsilon = self.epsilon - self.initial_epsilon*(self.epsilon_multiplier - 1)/4

            if success:
                count = 0
                past_qs.append(q)
            else:
                if uniform(0, 100, 1) < count:
                    if len(past_qs) > 0:
                        last_q, past_qs = past_qs[-1], past_qs[:-1]
                        delta = delta + self.epsilon * last_q
                        self.ps = self.ps[:-1]
                        count = 0

            if iteration % 100 == 0: # only save image and probs every 100 steps, to save memory space
                image_save = adv
                preds_save = top_preds
            else:
                image_save = None
                preds_save = None
                
            self.logger.append({
                "iterations": iteration,
                "epsilon": self.epsilon,
                "size": self.size,
                "is_adv": is_adv,
                "image": image_save,
                "top_preds": preds_save,
                "success": success,
            }, image, adv)

            # check if image is now adversarial
            if ((not is_adv) and (self.is_adversarial(top_preds[0][0], loss_label))):
                is_adv = 1
                self.logger.append({
                    "iterations": iteration,
                    "epsilon": self.epsilon,
                    "size": self.size,
                    "is_adv": is_adv,
                    "image": adv,
                    "top_preds": top_preds,
                    "success": success,
                }, image, adv) 
                return adv
                
        return None

    def check_pos(self, x, delta, q, p, loss_label):
        success = False 
        pos_x = x + delta + self.epsilon * q
        pos_x = clip(pos_x, self.bounds[0], self.bounds[1])
        top_5_preds = self.model.get_top_5(pos_x)
        if self.model.metadata['activation bits'] <= 8:
            noisy_top_5_preds = self.adjust_preds(top_5_preds)

        idx = argwhere(loss_label==top_5_preds[:,0]) # positions of occurences of label in preds
        if len(idx) == 0:
            print("{} does not appear in top_preds".format(loss_label))
            delta = delta - self.epsilon*q # add new perturbation to total perturbation
            success = True
            return delta, p, top_5_preds, success
        idx = idx[0][0]
        if self.model.metadata['activation bits'] <= 8:
            p_test = noisy_top_5_preds[idx][1]
        else:
            p_test = top_5_preds[idx][1]
        if p_test < self.ps[-1] or idx != 0:
            delta = delta + self.epsilon*q # add new perturbation to total perturbation
            self.ps.append(p_test) # update new p
            success = True
        return delta, p, top_5_preds, success

    def check_neg(self, x, delta, q, p, loss_label):
        success = False
        neg_x = x + delta - self.epsilon * q
        neg_x = clip(neg_x, self.bounds[0], self.bounds[1])
        top_5_preds = self.model.get_top_5(neg_x)
        if self.model.metadata['activation bits'] <= 8:
            noisy_top_5_preds = self.adjust_preds(top_5_preds)

        idx = argwhere(loss_label==top_5_preds[:,0]) # positions of occurences of label in preds
        if len(idx) == 0:
            print("{} does not appear in top_preds".format(loss_label))
            delta = delta - self.epsilon*q # add new perturbation to total perturbation
            success = True
            return delta, p, top_5_preds, success
        idx = idx[0][0]
        if self.model.metadata['activation bits'] <= 8:
            p_test = noisy_top_5_preds[idx][1]
        else:
            p_test = top_5_preds[idx][1]
        if p_test < self.ps[-1] or idx != 0:
            delta = delta - self.epsilon*q # add new perturbation to total perturbation
            self.ps.append(p_test) # update new p 
            success = True
        return delta, p, top_5_preds, success

    def is_adversarial(self, top_1_label, original_label):
        return top_1_label != original_label

    def new_q_direction(self, done):
        q_indicies = self.sample_nums(done)
        q = zeros((self.height, self.width, self.channels))
        for [a, b, c] in q_indicies:
            q = q + self.q_direction(a, b, c)
        return q, done

    def q_direction(self, a, b, c):
        q = zeros((self.height, self.width, self.channels))
        for i in range(self.size):
            for j in range(self.size):
                q[a*self.size+i, b*self.size+j, c] = 1
        q = q/norm(q)
        return q

    def adjust_preds(self, preds):
        probs = list(map(lambda x: x[1] + uniform(low=-0.000005, high=0.000005, size=1), preds))
        preds = list(map(lambda x: x[0], preds))
        return array(list(map(lambda x: array(x), zip(preds, probs))))

    def sample_nums(self, done):
        # samples new pixels without replacement
        indicies = []
        for _ in range(self.num_directions):
            [a, b, c] = self.sample_nums_rec()
            self.done.append([a, b, c])
            indicies.append([a, b, c])
            if len(self.done) >= self.height*self.width*self.channels/self.size/self.size-2:
                self.done = [] #empty it before it hits recursion limit
        return indicies

    def sample_nums_rec(self):
        [a,b] = randint(0, high=self.height/self.size, size=2)
        c = randint(0, high=self.channels, size=1)[0]
        if [a,b,c] in self.done:
            [a,b,c] = self.sample_nums_rec()
        return [a,b,c]
