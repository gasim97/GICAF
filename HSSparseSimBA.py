from gicaf.interface.AttackInterface import AttackInterface
import gicaf.Stats as stats
from sys import setrecursionlimit
from numpy import clip, argwhere, zeros, array, log, full, gradient, flip, dot, shape, linspace, empty, inf
from numpy.linalg import norm
from numpy.random import randint
from scipy.special import softmax
from math import ceil
import time
from logging import info

class HSSparseSimBA(AttackInterface):

    def __init__(self, size=1, epsilon=64, query_limit=5000): 
        self.size = size
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.query_limit = query_limit

    def run(self, image, model, logger):
        self.model = model
        self.height = self.model.metadata['height']
        self.width = self.model.metadata['width']
        self.channels = self.model.metadata['channels']
        self.bounds = self.model.metadata['bounds']
        self.logger = logger

        setrecursionlimit(max(1000, int(self.height*self.width*self.channels/self.size/self.size*2))) #for deep recursion diretion sampling
        
        top_preds = self.model.get_top_5(image)

        ####
        loss_label, p = top_preds[0]

        self.calcHS(image, loss_label)
        top_preds = self.get_top_5(image)
        _, p = top_preds[0]

        self.ps = [p]
        count = 0
        self.done = []
        self.num_directions = 1

        self.logger.nl(['iterations','total calls',
                        'epsilon','size', 'is_adv',
                        'ssim', 'psnr', 'wadiqam', 'normzero', 'image', 'top_preds', 'success'])
        self.total_calls = 0
        delta = 0
        is_adv = 0
        iteration = 0
        done = []
        
        #save step 0 in df
        adv = clip(image + delta, self.bounds[0], self.bounds[1])
        ssim = stats.ssim(image, adv, self.model.metadata)
        psnr = stats.psnr(image, adv, self.model.metadata)
        wadiqam = stats.wadiqam(image, adv, self.model.metadata)
        normZero = stats.normZero(image, adv, self.model.metadata)
        self.logger.append({
            "iterations": iteration,
            "total calls": self.total_calls,
            "epsilon": self.epsilon,
            "size": self.size,
            "is_adv": is_adv,
            "ssim": ssim,
            "psnr": psnr,
            "wadiqam": wadiqam,
            "normzero": normZero,
            "image": image,
            "top_preds": top_preds,
            "success": False,
        })

        while ((not is_adv) & (self.total_calls <= self.query_limit)): #buffer of 5 calls
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

            #update data on df
            adv = clip(image + delta, self.bounds[0], self.bounds[1])
            ssim = stats.ssim(image, adv, self.model.metadata)
            psnr = stats.psnr(image, adv, self.model.metadata)
            wadiqam = stats.wadiqam(image, adv, self.model.metadata)
            normZero = stats.normZero(image, adv, self.model.metadata)

            if iteration % 100 == 0: #only save image and probs every 100 steps, to save memory space
                image_save = adv
                preds_save = top_preds
            else:
                image_save = None
                preds_save = None
                
            self.logger.append({
                "iterations": iteration,
                "total calls": self.total_calls,
                "epsilon": self.epsilon,
                "size": self.size,
                "is_adv": is_adv,
                "ssim": ssim,
                "psnr": psnr,
                "wadiqam": wadiqam,
                "normzero": normZero,
                "image": image_save,
                "top_preds": preds_save,
                "success": success,
            })

            #check if image is now adversarial
            if ((not is_adv) and (self.is_adversarial(top_preds[0][0], loss_label))):
                is_adv = 1
                self.logger.append({
                    "iterations": iteration,
                    "total calls": self.total_calls,
                    "epsilon": self.epsilon,
                    "size": self.size,
                    "is_adv": is_adv,
                    "ssim": ssim,
                    "psnr": psnr,
                    "wadiqam": wadiqam,
                    "normzero": normZero,
                    "image": adv,
                    "top_preds": top_preds,
                    "success": success,
                }) 
                return adv #remove this to continue attack even after adversarial is found
                
        return None

    # def hessian(self, x):
    #     """
    #     Calculate the hessian matrix with finite differences
    #     Parameters:
    #     - x : ndarray
    #     Returns:
    #     an array of shape (x.dim, x.ndim) + x.shape
    #     where the array[i, j, ...] corresponds to the second derivative x_ij
    #     """
    #     x_grad = gradient(x) 
    #     hessian = empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
    #     for k, grad_k in enumerate(x_grad):
    #         # iterate over dimensions
    #         # apply gradient again to every component of the first derivative.
    #         tmp_grad = gradient(grad_k) 
    #         for l, grad_kl in enumerate(tmp_grad):
    #             hessian[k, l, :, :] = grad_kl
    #     return hessian

    # def calcHSFull(self, image, label):
    #     preds = empty((self.height, self.width, self.channels))
    #     y = array(list(map(lambda i: 0 if i != label else 1, range(len(preds[0])))))
    #     for a in range(self.height):
    #         for b in range(self.width):
    #             for c in range(self.channels):
    #                 q = self.q_direction(a, b, c)
    #                 x = clip(image + (self.epsilon * q), self.bounds[0], self.bounds[1])
    #                 preds[a][b][c] = array(list(map(lambda p: p[1], self.model.get_preds(x))))

    #     betas = linspace(0.1, 2, 20)
    #     max_norm = 0
    #     for beta in betas:
    #         losses = empty((self.height, self.width, self.channels))
    #         for a in range(self.height):
    #             for b in range(self.width):
    #                 for c in range(self.channels):
    #                     losses[a][b][c] = dot(-y.T, log(softmax(beta*log(preds[a][b][c]))))
    #         hessian_norm = norm(self.hessian(losses))
    #         print('beta, hessian norm: ' + str(beta) + ', ' + str(hessian_norm))
    #         if (hessian_norm > max_norm):
    #             max_norm = hessian_norm
    #             self.beta = beta
    #     print('Selected beta: ' + str(self.beta))

    def calcHS(self, image, label):
        delta = full([self.height, self.width, self.channels], self.epsilon/(self.height*self.width*self.channels))
        x_pos = clip(image + delta, self.bounds[0], self.bounds[1])
        x_pos_2 = clip(x_pos + delta, self.bounds[0], self.bounds[1])
        x_neg = clip(image - delta, self.bounds[0], self.bounds[1])
        x_neg_2 = clip(x_neg - delta, self.bounds[0], self.bounds[1])
        preds = [array(list(map(lambda p: p[1], self.model.get_preds(x_neg_2)))), 
                  array(list(map(lambda p: p[1], self.model.get_preds(x_neg)))), 
                  array(list(map(lambda p: p[1], self.model.get_preds(image)))), 
                  array(list(map(lambda p: p[1], self.model.get_preds(x_pos)))),
                  array(list(map(lambda p: p[1], self.model.get_preds(x_pos_2))))] # map to extract probability in position 1 of each element and throw away label

        y = array(list(map(lambda i: 0 if i != label else 1, range(len(preds[0])))))
        betas = linspace(0.1, 2, 20)
        max_norm = 0
        for beta in betas:
            loss = [dot(-y.T, log(softmax(beta*log(preds[0])))), dot(-y.T, log(softmax(beta*log(preds[1])))), dot(-y.T, log(softmax(beta*log(preds[2]))))]
            hessian_norm = norm(gradient(gradient(loss)))
            print('beta, hessian norm: ' + str(beta) + ', ' + str(hessian_norm))
            if (hessian_norm > max_norm):
                max_norm = hessian_norm
                self.beta = beta
        print('Selected beta: ' + str(self.beta))

    def get_top_5(self, x):
        preds = self.model.get_preds(x)
        probs = softmax(self.beta*log(array(list(map(lambda x: x[1], preds)))))
        preds = array(list(map(lambda x: array([int(x[0]), probs[int(x[0])]]), preds)))
        return flip(preds[preds[:, 1].argsort()][-5:], 0)

    def check_pos(self, x, delta, q, p, loss_label):
        success = False #initialise as False by default
        pos_x = x + delta + self.epsilon * q
        pos_x = clip(pos_x, self.bounds[0], self.bounds[1])
        top_5_preds = self.get_top_5(pos_x)
        if self.model.metadata['activation_bits'] <= 8:
            top_5_preds = self.adjust_preds(top_5_preds)

        idx = argwhere(loss_label==top_5_preds[:,0]) #positions of occurences of label in preds
        if len(idx) == 0:
            print("{} does not appear in top_preds".format(loss_label))
            delta = delta - self.epsilon*q #add new perturbation to total perturbation
            success = True
            return delta, p, top_5_preds, success
        idx = idx[0][0]
        p_test = top_5_preds[idx][1]
        if p_test < self.ps[-1] or idx != 0:
            delta = delta + self.epsilon*q #add new perturbation to total perturbation
            self.ps.append(p_test) #update new p
            success = True
        return delta, p, top_5_preds, success

    def check_neg(self, x, delta, q, p, loss_label):
        success = False #initialise as False by default
        neg_x = x + delta - self.epsilon * q
        # neg_x = self.adjust_pertubation(neg_x, q_indicies)
        neg_x = clip(neg_x, self.bounds[0], self.bounds[1])
        top_5_preds = self.get_top_5(neg_x)
        if self.model.metadata['activation_bits'] <= 8:
            top_5_preds = self.adjust_preds(top_5_preds)

        idx = argwhere(loss_label==top_5_preds[:,0]) #positions of occurences of label in preds
        if len(idx) == 0:
            print("{} does not appear in top_preds".format(loss_label))
            delta = delta - self.epsilon*q #add new perturbation to total perturbation
            success = True
            return delta, p, top_5_preds, success
        idx = idx[0][0]
        p_test = top_5_preds[idx][1]
        if p_test < self.ps[-1] or idx != 0:
            delta = delta - self.epsilon*q #add new perturbation to total perturbation
            self.ps.append(p_test) #update new p 
            success = True
        return delta, p, top_5_preds, success

    def is_adversarial(self, top_1_label, original_label):
        #returns whether image is adversarial, according to setting
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
        probs = list(map(lambda x: round(x[1], 5), preds))
        preds = list(map(lambda x: x[0], preds))
        return array(list(map(lambda x: array(x), zip(preds, probs))))

    def sample_nums(self, done):
        #samples new pixels without replacement
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
            #sample again (recursion)
            [a,b,c] = self.sample_nums_rec()
        return [a,b,c]

    def close(self):
        pass