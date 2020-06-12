"""
    Initial Implementation by Alvaro Robledo
------
Sourced and adapted from: https://github.com/alvarorobledo/sparse-simba/blob/master/utils.py
Obtained: 27/01/2020
"""
from typing import Type, Optional
from gicaf.interface.AttackBase import AttackBase
from gicaf.interface.ModelBase import ModelBase
from gicaf.interface.LoggerBase import LoggerBase
from numpy import clip, argwhere, zeros, ndarray
from sys import setrecursionlimit
from numpy.linalg import norm
from numpy.random import randint
import time

class SparseSimBA(AttackBase):

    def __init__(
        self, 
        size: int = 1, 
        epsilon: int = 64, 
    ) -> None: 
        self.size = size
        self.epsilon = epsilon

    def __call__(self, 
        image: ndarray, 
        model: Type[ModelBase], 
        logger: Type[LoggerBase],
        ground_truth: Optional[int] = None, 
        target: Optional[int] = None, 
        query_limit: int = 5000
    ) -> Optional[ndarray]:
        if target:
            raise NotImplementedError("Targeted Sparse SimBA has not been implemented yet") 
        if ground_truth == None:
            raise ValueError('Sparse SimBA is not intended for generating false positives, please provide a ground truth')
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

        ####
        top_1_label, p = top_preds[0]
        self.logger.nl(['iterations', 'epsilon','size', 
                        'is_adv', 'image', 'top_preds'])
        total_calls = 0
        delta = 0
        is_adv = self.is_adversarial(top_1_label, loss_label)
        iteration = 0
        done = []
        
        #save step 0 in df
        adv = clip(image + delta, self.bounds[0], self.bounds[1])

        self.logger.append({
            "iterations": iteration,
            "epsilon": self.epsilon,
            "size": self.size,
            "is_adv": is_adv,
            "image": image,
            "top_preds": top_preds
        }, image, adv)

        while ((not is_adv) & (self.model.get_query_count() <= self.query_limit)):
            iteration += 1    

            q, done = self.new_q_direction(done)

            delta, p, top_preds, success = self.check_pos(image, delta, q, p, loss_label)
            total_calls += 1
            if not success:
                delta, p, top_preds, _ = self.check_neg(image, delta, q, p, loss_label)
                total_calls += 1

            #update data on df
            adv = clip(image + delta, self.bounds[0], self.bounds[1])

            if iteration % 100 == 0: #only save image and probs every 100 steps, to save memory space
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
                "top_preds": preds_save
            }, image, adv)

            #check if image is now adversarial
            if ((not is_adv) and (self.is_adversarial(top_preds[0][0], loss_label))):
                is_adv = 1
                self.logger.append({
                    "iterations": iteration,
                    "epsilon": self.epsilon,
                    "size": self.size,
                    "is_adv": is_adv,
                    "image": adv,
                    "top_preds": top_preds
                }, image, adv) 
                return adv #remove this to continue attack even after adversarial is found
                
        return None

    def check_pos(self, x, delta, q, p, loss_label):
        success = False #initialise as False by default
        pos_x = x + delta + self.epsilon * q
        pos_x = clip(pos_x, self.bounds[0], self.bounds[1])
        top_5_preds = self.model.get_top_5(pos_x)

        idx = argwhere(loss_label==top_5_preds[:,0]) #positions of occurences of label in preds
        if len(idx) == 0:
            print("{} does not appear in top_preds".format(loss_label))
            return delta, p, top_5_preds, success
        idx = idx[0][0]
        p_test = top_5_preds[idx][1]
        if p_test < p:
            delta = delta + self.epsilon*q #add new perturbation to total perturbation
            p = p_test #update new p
            success = True
        return delta, p, top_5_preds, success

    def check_neg(self, x, delta, q, p, loss_label):
        success = False #initialise as False by default
        neg_x = x + delta - self.epsilon * q
        neg_x = clip(neg_x, self.bounds[0], self.bounds[1])
        top_5_preds = self.model.get_top_5(neg_x)

        idx = argwhere(loss_label==top_5_preds[:,0]) #positions of occurences of label in preds
        if len(idx) == 0:
            print("{} does not appear in top_preds".format(loss_label))
            return delta, p, top_5_preds, success
        idx = idx[0][0]
        p_test = top_5_preds[idx][1]
        if p_test < p:
            delta = delta - self.epsilon*q #add new perturbation to total perturbation
            p = p_test #update new p 
            success = True
        return delta, p, top_5_preds, success

    def is_adversarial(self, top_1_label, original_label):
        #returns whether image is adversarial, according to setting
        return top_1_label != original_label

    def new_q_direction(self, done):
        [a,b,c] = self.sample_nums(done)
        done.append([a,b,c])
        if len(done) >= self.height*self.width*self.channels/self.size/self.size-2:
            done = [] #empty it before it hits recursion limit
        q = zeros((self.height, self.width, self.channels))
        for i in range(self.size):
            for j in range(self.size):
                q[a*self.size+i, b*self.size+j, c] = 1
        q = q/norm(q)
        return q, done

    def sample_nums(self, done):
        #samples new pixels without replacement
        [a,b] = randint(0, high=self.height/self.size, size=2)
        c = randint(0, high=self.channels, size=1)[0]
        if [a,b,c] in done:
            #sample again (recursion)
            [a,b,c] = self.sample_nums(done)
        return [a,b,c]
