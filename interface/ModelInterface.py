from abc import ABCMeta, abstractmethod
from numpy import array, arange, flip, expand_dims, float32
import tensorflow.lite as lite
from copy import deepcopy

class ModelInterface:
    __metaclass__ = ABCMeta

    @classmethod
    def version(cls): return "1.0"

    @abstractmethod
    def __init__(self, model, metadata): 
        """
        Initialize model

        Parameters
        ----------
            model : Any
                A model of the type expected by the inheriting class
            metadata: dict
                Contains the following fields
                    'height' : int
                        Input height
                    'width' : int
                        Input width
                    'channels' : int
                        Input number of channels
                    'bounds' : 2-tuple with elements of type int
                        Contains (min, max) of input values
                    'bgr' : bool
                        True if input is to be loaded in BGR format, and False if RGB
                    'classes': int
                        Number of output classes
                    'weight_bits': int
                        Model weights bit width
                    'activation_bits': int
                        Model activations bit width
                    
        """
        raise NotImplementedError("Model module __init__() function missing")

    @abstractmethod
    def get_preds(self, image):
        """
        Run inference

        Parameters
        ----------
            image : numpy.ndarray
                Image
        Returns
        -------
            [[label, probability]] : numpy.ndarray -> shape = (number of classes, 2)
                The output indicies zipped with the predictions
        """
        raise NotImplementedError("Model module get_preds() function missing")

    @abstractmethod
    def get_preds_batch(self, images): 
        """
        Run inference on batch

        Parameters
        ----------
            images : numpy.ndarray or list with elements (images) of type numpy.ndarrays
                Images
        Returns
        -------
            [[[label, probability]]] : numpy.ndarray -> shape = (batch size, number of classes, 2)
                The output indicies zipped with the predictions
        """
        raise NotImplementedError("Model module get_preds_batch() function missing")

    @abstractmethod
    def close(self):
        """
        End of session clean up
        """
        pass

    def get_top_1(self, image): 
        """
        Run inference and return top 1

        Parameters
        ----------
            image : numpy.ndarray
                Image
        Returns
        -------
            label : int
                The output index corresponding to the highest prediction
            probability : float
                The highest output probability/value
        """
        preds = self.get_preds(image)
        return preds[preds[:, 1].argsort()][-1] # sort predictions by probability, then extract the last entry (highest probability with label)

    def get_top_1_batch(self, images): 
        """
        Run inference on batch and return top 1

        Parameters
        ----------
            images : numpy.ndarray or list with elements (images) of type numpy.ndarrays
                Images
        Returns
        -------
            [[label, probability]] : numpy.ndarray -> shape = (batch size, 2)
                The output indicies corresponding to the highest predictions
                for each image zipped with the predictions
        """
        preds = self.get_preds_batch(images)
        return array(list(map(lambda x: x[x[:, 1].argsort()][-1], preds)))

    def get_top_5(self, image): 
        """
        Run inference and return top 5 ORDERED HIGHEST TO LOWEST

        Parameters
        ----------
            images : numpy.ndarray or list with elements (images) of type numpy.ndarrays
                Images
        Returns
        -------
            [[label, probability]]  : numpy.ndarray -> shape = (5, 2)
                The output indicies corresponding to the highest 5 predictions
                zipped with the predictions
        """
        preds = self.get_preds(image)
        return flip(preds[preds[:, 1].argsort()][-5:], 0) # sort predictions by probability, then extract the last 5 entries (highest probabilities with labels) and flip to sort by descending probability

    def get_top_5_batch(self, images): 
        """
        Run inference on batch and return top 5 ORDERED HIGHEST TO LOWEST

        Parameters
        ----------
            images : numpy.ndarray or list with elements (images) of type numpy.ndarrays
                Images
        Returns
        -------
            [[[label, probability]]] : numpy.ndarray -> shape = (batch size, 5, 2)
                The output indicies corresponding to the highest 5 predictions
                for each image zipped with the predictions
        """
        preds = self.get_preds_batch(images)
        return flip(array(list(map(lambda x: x[x[:, 1].argsort()][-5:], preds))), 1)

    def zip_labels_to_probs(self, probs):
        """
        Zips the predictions with their indicies for convenience

        Parameters
        ----------
            probs : numpy.ndarray or list
                Predictions
        Returns
        -------
            [[label, probability]] : numpy.ndarray -> shape = (number of classes, 2)
                The predictions zipped with their indicies
        """
        return array(list(zip(arange(len(probs)), probs)))

class KerasModel(ModelInterface):

    def __init__(self, kmodel, metadata): 
        self.metadata = metadata
        self.model = kmodel

    def get_preds(self, image):
        return self.zip_labels_to_probs(self.model.predict(image))

    def get_preds_batch(self, images): 
        return array(list(map(lambda x: self.zip_labels_to_probs(x), self.model.predict(images))))

class TfLiteModel(ModelInterface):

    def __init__(self, interpreter, metadata): 
        self.metadata = metadata
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]

    def _evaluate(self, image):
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

    def get_preds(self, image):
        return self.zip_labels_to_probs(self._evaluate(image))

    def get_preds_batch(self, images): 
        return array(list(map(lambda img: self.zip_labels_to_probs(self._evaluate(img)), images)))

class PyTorchModel(ModelInterface):

    def __init__(self, model, metadata): 
        self.metadata = metadata
        self.model = model
        self.model.eval()

    def get_preds(self, image):
        return self.zip_labels_to_probs(self.model([image]).detach().numpy()[0])

    def get_preds_batch(self, images): 
        return array(list(map(lambda x: self.zip_labels_to_probs(x), self.model(images).detach().numpy())))
