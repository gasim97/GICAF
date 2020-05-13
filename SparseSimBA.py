from gicaf.interface.AttackInterface import AttackInterface
import gicaf.Stats as stats
from sys import setrecursionlimit
from numpy import clip, argwhere, zeros
from numpy.linalg import norm
from numpy.random import randint
from skimage.measure import compare_ssim, compare_psnr
from pandas import DataFrame
import time
from logging import info

class SparseSimBA(AttackInterface):

    # initialize
    def __init__(self): 
        self.size = 1
        self.epsilon = 64

    # runs the attack
    def run(self, images, model, logger, query_limit=5000, dims=(224, 224, 3), bounds=(0.0, 1.0)): 
        self.model = model
        self.logger = logger
        self.bounds = bounds
        self.dims = dims
        for image in images:
            self.run_sparse_simba(image, query_limit=query_limit)
        # print("Attack module run() function missing")
        # raise NotImplementedError
    # returns adversarial images, attack log


    def run_sparse_simba(self, image, epsilon=64, query_limit=5000, log_every_n_steps=200):
        setrecursionlimit(max(1000, int(self.model.metadata()['height']*self.model.metadata()['width']*self.model.metadata()['channels']/self.size/self.size))) #for deep recursion diretion sampling
        top_preds = self.model.get_top_5(image)

        ####
        loss_label, p = top_preds[0]
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
        psnr = stats.psnr(image, adv, data_range=255)
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

            q, done = self.new_q_direction(done, size=self.size)

            delta, p, top_preds, success = self.check_pos(image, delta, q, p, loss_label)
            total_calls += 1
            if not success:
                delta, p, top_preds, _ = self.check_neg(image, delta, q, p, loss_label)
                total_calls += 1

            #update data on df
            adv = clip(image + delta, self.bounds[0], self.bounds[1])
            ssim = stats.ssim(image, adv, multichannel=True)
            psnr = stats.psnr(image, adv, data_range=255)

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

    def new_q_direction(self, done, size=1):
        [a,b,c] = self.sample_nums(done, size)
        done.append([a,b,c])
        if len(done) >= self.dims[0]*self.dims[1]*self.dims[2]/size/size-2:
            done = [] #empty it before it hits recursion limit
        q = zeros((self.dims[0],self.dims[1],self.dims[2]))
        for i in range(size):
            for j in range(size):
                q[a*size+i, b*size+j, c] = 1
        q = q/norm(q)
        return q, done

    def sample_nums(self, done, size=1):
        #samples new pixels without replacement
        [a,b] = randint(0, high=self.dims[1]/size, size=2)
        c = randint(0, high=3, size=1)[0]
        if [a,b,c] in done:
            #sample again (recursion)
            [a,b,c] = self.sample_nums(done, size)
        return [a,b,c]