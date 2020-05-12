from abc import ABCMeta, abstractmethod
from numpy import array, arange, flip
from foolbox.models import KerasModel

class ModelInterface:
    __metaclass__ = ABCMeta

    @classmethod
    def version(cls): return "1.0"

    # initialize
    @abstractmethod
    def __init__(self): 
        print("Model module __init__() function missing")
        raise NotImplementedError

    # get model metadata
    @abstractmethod
    def metadata(self): 
        print("Model module get_model_meta() function missing")
        raise NotImplementedError
    # returns input height, input width, input channels, (True if BGR else False)

    # run inference and return top 1
    @abstractmethod
    def get_top_1(self, image): 
        print("Model module get_top_1() function missing")
        raise NotImplementedError
    # returns [label, probability] -> shape = (1, 2)

    # run inference on batch and return top 1
    @abstractmethod
    def get_top_1_batch(self, images): 
        print("Model module get_top_1_batch() function missing")
        raise NotImplementedError
    # returns [[label, probability]] -> shape = (batch size, 2)

    # run inference and return top 5 ORDERED HIGHEST TO LOWEST
    @abstractmethod
    def get_top_5(self, image): 
        print("Model module get_top_5() function missing")
        raise NotImplementedError
    # returns [[label, probability]] -> shape = (5, 2)

    # run inference on batch and return top 5 ORDERED HIGHEST TO LOWEST
    @abstractmethod
    def get_top_5_batch(self, images): 
        print("Model module get_top_5_batch() function missing")
        raise NotImplementedError
    # returns [[[label, probability]]] -> shape = (batch size, 5, 2)

    # end of session clean up
    @abstractmethod
    def close(self):
        pass

class FoolboxKerasModelInterface(ModelInterface):

    # initialize
    def __init__(self, kmodel, preprocessing=(0, 1)): 
        #sets up local model, to use for local testing
        self.model = KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing, predicts='logits')

    def zip_labels_to_probs(self, probs):
        return array(list(zip(arange(len(probs)), probs)))

    # run inference
    def get_preds(self, image):
        return self.zip_labels_to_probs(self.model.predictions(image))

    # run inference on batch
    def get_preds_batch(self, images): 
        return array(list(map(lambda x: self.zip_labels_to_probs(x), self.model.batch_predictions(images))))

    # run inference and return top 1
    def get_top_1(self, image): 
        preds = self.get_preds(image)
        return preds[preds[:, 1].argsort()][-1] # sort predictions by probability, then extract the last entry (highest probability with label)
    # returns [label, probability]

    # run inference on batch and return top 1
    def get_top_1_batch(self, images): 
        preds = self.get_preds_batch(images)
        return array(list(map(lambda x: x[x[:, 1].argsort()][-1], preds)))
    # returns [[label, probability]]

    # run inference and return top 5 ORDERED HIGHEST TO LOWEST
    def get_top_5(self, image): 
        preds = self.get_preds(image)
        return flip(preds[preds[:, 1].argsort()][-5:], 0) # sort predictions by probability, then extract the last 5 entries (highest probabilities with labels) and flip to sort by descending probability
    # returns [[label, probability]] 

    # run inference on batch and return top 5 ORDERED HIGHEST TO LOWEST
    def get_top_5_batch(self, images): 
        preds = self.get_preds_batch(images)
        return flip(array(list(map(lambda x: x[x[:, 1].argsort()][-5:], preds))), 1)
    # returns [[[label, probability]]]

class PyTorchModel(ModelInterface):

    # initialize
    @abstractmethod
    def __init__(self): 
        print("Model module __init__() function missing")
        raise NotImplementedError

    def zip_labels_to_probs(self, probs):
        return array(list(zip(arange(len(probs)), probs)))

    # run inference
    def get_preds(self, image):
        return self.zip_labels_to_probs(self.model.predictions(image))

    # run inference on batch
    def get_preds_batch(self, images): 
        return array(list(map(lambda x: self.zip_labels_to_probs(x), self.model.batch_predictions(images))))

    # run inference and return top 1
    def get_top_1(self, image): 
        preds = self.get_preds(image)
        return preds[preds[:, 1].argsort()][-1] # sort predictions by probability, then extract the last entry (highest probability with label)
    # returns [label, probability]

    # run inference on batch and return top 1
    def get_top_1_batch(self, images): 
        preds = self.get_preds_batch(images)
        return array(list(map(lambda x: x[x[:, 1].argsort()][-1], preds)))
    # returns [[label, probability]]

    # run inference and return top 5 ORDERED HIGHEST TO LOWEST
    def get_top_5(self, image): 
        preds = self.get_preds(image)
        return flip(preds[preds[:, 1].argsort()][-5:], 0) # sort predictions by probability, then extract the last 5 entries (highest probabilities with labels) and flip to sort by descending probability
    # returns [[label, probability]] 

    # run inference on batch and return top 5 ORDERED HIGHEST TO LOWEST
    def get_top_5_batch(self, images): 
        preds = self.get_preds_batch(images)
        return flip(array(list(map(lambda x: x[x[:, 1].argsort()][-5:], preds))), 1)
    # returns [[[label, probability]]]
