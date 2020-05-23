from gicaf.interface.AttackInterface import AttackInterface
import gicaf.Stats as stats
from sys import setrecursionlimit
from numpy import clip, argwhere, zeros, array, log, full, gradient, flip
from numpy.linalg import norm
from numpy.random import randint
from scipy.special import softmax
import time
from logging import info

class HSSparseSimBA(AttackInterface):

    def __init__(self, size=1, epsilon=64): 
        self.size = size
        self.epsilon = epsilon

    # runs the attack
    def run(self, images, model, logger, query_limit=5000, bounds=(0.0, 1.0)): 
        self.model = model
        self.height = self.model.metadata()['height']
        self.width = self.model.metadata()['width']
        self.channels = self.model.metadata()['channels']
        self.logger = logger
        self.bounds = bounds
        for image in images:
            self.run_sparse_simba(image, query_limit=query_limit)
    # returns adversarial images, attack log

    def run_sparse_simba(self, image, query_limit=5000, log_every_n_steps=200):
        setrecursionlimit(max(1000, int(self.height*self.width*self.channels/self.size/self.size))) #for deep recursion diretion sampling
        top_preds = self.model.get_top_5(image)

        ####
        loss_label, p = top_preds[0]
        self.calcHS(image, loss_label)
        self.logger.nl(['iterations','total calls',
                        'epsilon','size', 'is_adv',
                        'ssim', 'psnr', 'image', 'top_preds'])
        total_calls = 0
        delta = 0
        is_adv = 0
        iteration = 0
        done = []
        
        #save step 0 in df
        adv = clip(image + delta, self.bounds[0], self.bounds[1])
        ssim = stats.ssim(image, adv, multichannel=True)
        psnr = stats.psnr(image, adv, data_range=(self.bounds[1] - self.bounds[0]))
        self.logger.append({
            "iterations": iteration,
            "total calls": total_calls,
            "epsilon": self.epsilon,
            "size": self.size,
            "is_adv": is_adv,
            "ssim": ssim,
            "psnr": psnr,
            "image": image,
            "top_preds": top_preds
        })

        start = time.time()

        while ((not is_adv) & (total_calls <= query_limit+5)): #buffer of 5 calls
            if iteration % log_every_n_steps == 0:
                print('iteration: {}, new p is: {}, took {:.2f} s'.format(str(iteration), str(p), time.time()-start))
            iteration += 1    

            q, done = self.new_q_direction(done)

            delta, p, top_preds, success = self.check_pos(image, delta, q, p, loss_label)
            total_calls += 1
            if not success:
                delta, p, top_preds, _ = self.check_neg(image, delta, q, p, loss_label)
                total_calls += 1

            #update data on df
            adv = clip(image + delta, self.bounds[0], self.bounds[1])
            ssim = stats.ssim(image, adv, multichannel=True)
            psnr = stats.psnr(image, adv, data_range=(self.bounds[1] - self.bounds[0]))

            if iteration % 100 == 0: #only save image and probs every 100 steps, to save memory space
                image_save = adv
                preds_save = top_preds
            else:
                image_save = None
                preds_save = None
                
            self.logger.append({
                "iterations": iteration,
                "total calls": total_calls,
                "epsilon": self.epsilon,
                "size": self.size,
                "is_adv": is_adv,
                "ssim": ssim,
                "psnr": psnr,
                "image": image_save,
                "top_preds": preds_save
            })

            #check if image is now adversarial
            if ((not is_adv) and (self.is_adversarial(top_preds[0][0], loss_label))):
                is_adv = 1
                self.logger.append({
                    "iterations": iteration,
                    "total calls": total_calls,
                    "epsilon": self.epsilon,
                    "size": self.size,
                    "is_adv": is_adv,
                    "ssim": ssim,
                    "psnr": psnr,
                    "image": adv,
                    "top_preds": top_preds
                }) 
                return adv, total_calls #remove this to continue attack even after adversarial is found
                
        return adv, total_calls

    def calcHS(self, image, label):
        delta = full([self.height, self.width, self.channels], self.epsilon/self.channels)
        x_pos = clip(image + delta, self.bounds[0], self.bounds[1])
        x_neg = clip(image - delta, self.bounds[0], self.bounds[1])
        preds = [array(list(map(lambda p: p[1], self.model.get_preds(x_neg)))), array(list(map(lambda p: p[1], self.model.get_preds(image)))), array(list(map(lambda p: p[1], self.model.get_preds(x_pos))))] # map to extract probability in position 1 of each element and throw away label

        y = array(list(map(lambda i: 0 if i != label else 1, len(preds))))
        betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
        max_norm = 0
        for beta in betas:
            loss = [-y.T*log(softmax(beta*preds[0])), -y.T*log(softmax(beta*preds[1])), -y.T*log(softmax(beta*preds[2]))]
            norm = gradient(gradient(loss))
            if (norm > max_norm):
                max_norm = norm
                self.beta = beta

    def get_top_5(self, x):
        preds = self.model.get_preds(x)
        probs = softmax(self.beta*list(map(lambda x: x[1], preds)))
        preds = array(list(map(lambda x: array([x[0], probs[x[0]]]), preds)))
        return flip(preds[preds[:, 1].argsort()][-5:], 0)

    def check_pos(self, x, delta, q, p, loss_label):
        success = False #initialise as False by default
        pos_x = x + delta + self.epsilon * q
        pos_x = clip(pos_x, self.bounds[0], self.bounds[1])
        top_5_preds = self.get_top_5(pos_x)

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
        top_5_preds = self.get_top_5(neg_x)

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