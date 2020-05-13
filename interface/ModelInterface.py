from abc import ABCMeta, abstractmethod
from numpy import array, arange, flip, expand_dims, float32
from foolbox.models import KerasModel
import tensorflow.lite as lite
from copy import deepcopy

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

    def zip_labels_to_probs(self, probs):
        return array(list(zip(arange(len(probs)), probs)))

class FoolboxKerasModelInterface(ModelInterface):

    # initialize
    def __init__(self, kmodel, preprocessing=(0, 1)): 
        #sets up local model, to use for local testing
        self.model = KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing, predicts='logits')

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

class TfLiteModel(ModelInterface):
    # initialize
    def __init__(self, tflite_model_file_path): 
        #sets up local model, to use for local testing
        self.interpreter = lite.Interpreter(model_path=str(tflite_model_file_path))
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]

    def evaluate(self, image):
        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.
        img = expand_dims(image, axis=0).astype(float32)
        self.interpreter.set_tensor(self.input_index, img)
        # Run inference.
        self.interpreter.invoke()
        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        output = self.interpreter.tensor(self.output_index)
        return deepcopy(output()[0])

    # run inference
    def get_preds(self, image):
        return self.zip_labels_to_probs(self.evaluate(image))

    # run inference on batch
    def get_preds_batch(self, images): 
        predictions = []
        for img in images:
            predictions.append(self.evaluate(img))
        return array(list(map(lambda x: self.zip_labels_to_probs(x), predictions)))

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
