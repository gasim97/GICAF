from typing import List, Union, Any, Mapping, Union, Tuple
from abc import ABC, abstractmethod
from numpy import array, arange, flip, expand_dims, float32, ndarray
from scipy.special import softmax
from copy import deepcopy

class ModelBase(ABC):

    @classmethod
    def version(cls): return "1.0"

    @classmethod
    def zip_indicies_to_preds(
        cls, 
        preds: Union[List, ndarray]
    ) -> ndarray:
        """
        Zips the predictions with their indicies for convenience

        Parameters
        ----------
            preds : numpy.ndarray or list
                Predictions
        Returns
        -------
            [[label, probability]] : numpy.ndarray -> shape = (number of classes, 2)
                The predictions zipped with their indicies
        """
        return array(list(zip(arange(len(preds)), preds)))

    @abstractmethod
    def __init__(
        self, 
        model: Any, 
        metadata: Mapping[str, Union[int, bool, Tuple[int, int]]]
    ) -> None: 
        """
        Initialize model

        Parameters
        ----------
            model : Any
                A model of the type expected by the inheriting class
            metadata: dict
                To be stored as an instance variable 'self.metadata' and contains the 
                following fields
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
                    'apply_softmax' : bool
                        Whether or not to apply softmax to the model's output
                    'weight_bits': int
                        Model weights bit width
                    'activation_bits': int
                        Model activations bit width
        Note
        ----
            This method must create the following variables:
                self.metadata = metadata
                self.query_count = 0
        """
        ...

    @abstractmethod
    def get_preds(
        self, 
        image: ndarray
    ) -> ndarray:
        """
        Run inference on the model and apply softmax to the output if 
        'apply_sfotmax' field in 'self.metadata' is True

        Parameters
        ----------
            image : numpy.ndarray
                Image
        Returns
        -------
            [[label, probability]] : numpy.ndarray -> shape = (number of classes, 2)
                The output indicies zipped with the predictions
        Note
        ----
            This method must call 'self.queries_count(1)' to increment the query count
        """
        ...

    @abstractmethod
    def get_preds_batch(
        self, 
        images: ndarray
    ) -> ndarray: 
        """
        Run inference on batch and apply softmax to the output if 
        'apply_sfotmax' field in 'self.metadata' is True

        Parameters
        ----------
            images : numpy.ndarray or list with elements (images) of type numpy.ndarrays
                Images
        Returns
        -------
            [[[label, probability]]] : numpy.ndarray -> shape = (batch size, number of classes, 2)
                The output indicies zipped with the predictions
        Note
        ----
            This method must call 'self.queries_count(len(images))' to increment the query count
        """
        ...

    def get_top_1(
        self, 
        image: ndarray
    ) -> ndarray: 
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
        return preds[preds[:, 1].argsort()][-1] # sort predictions by probability, then 
        # extract the last entry (highest probability with label)

    def get_top_1_batch(
        self, 
        images: ndarray
    ) -> ndarray: 
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

    def get_top_5(
        self, 
        image: ndarray
    ) -> ndarray: 
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
        return flip(preds[preds[:, 1].argsort()][-5:], 0) # sort predictions by probability, 
        # then extract the last 5 entries (highest probabilities with labels) and flip to sort 
        # by descending probability

    def get_top_5_batch(
        self, 
        images: ndarray
    ) -> ndarray: 
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

    def increment_query_count(
        self, 
        num_queries: int
    ) -> None:
        """
        Increases the model's internal query count

        Parameters
        ----------
            num_queries : int
                The number of queries to add to the current query count
        """
        self.query_count += num_queries

    def reset_query_count(self) -> None:
        """
        Resets the model's internal query count to zero
        """
        self.query_count = 0

    def get_query_count(self) -> int:
        return self.query_count

    def close(self) -> None:
        """
        End of session clean up
        """
        pass

class KerasModel(ModelBase):

    def __init__(
        self, 
        kmodel: Any, 
        metadata: Mapping[str, Union[int, bool, Tuple[int, int]]]
    ) -> None: 
        self.metadata = metadata
        self.query_count = 0
        self.model = kmodel

    def get_preds(
        self, 
        image: ndarray
    ) -> ndarray:
        preds = self.model.predict(image)
        self.increment_query_count(1)
        return ModelBase.zip_indicies_to_preds(preds if not self.metadata['apply_softmax'] else softmax(preds))

    def get_preds_batch(
        self, 
        images: ndarray
    ) -> ndarray: 
        preds = self.model.predict(images)
        self.increment_query_count(len(images))
        return array(list(map(
            lambda x: ModelBase.zip_indicies_to_preds(x), 
            preds if not self.metadata['apply_softmax'] else map(lambda x: softmax(x), preds))))

class TfLiteModel(ModelBase):

    def __init__(
        self, 
        interpreter: Any, 
        metadata: Mapping[str, Union[int, bool, Tuple[int, int]]]
    ) -> None: 
        self.metadata = metadata
        self.query_count = 0
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]

    def _evaluate(
        self, 
        image: ndarray
    ) -> List:
        img = expand_dims(image, axis=0).astype(float32)
        self.interpreter.set_tensor(self.input_index, img)
        self.interpreter.invoke()
        output = self.interpreter.tensor(self.output_index)
        return deepcopy(output()[0])

    def get_preds(
        self, 
        image: ndarray
    ) -> ndarray:
        preds = self._evaluate(image)
        self.increment_query_count(1)
        return ModelBase.zip_indicies_to_preds(preds if not self.metadata['apply_softmax'] else softmax(preds))

    def get_preds_batch(
        self, 
        images: ndarray
    ) -> ndarray: 
        self.increment_query_count(len(images))
        return array(list(map(
            lambda img: ModelBase.zip_indicies_to_preds(self._evaluate(img) if not self.metadata['apply_softmax'] else softmax(self._evaluate(img))), 
            images)))

class PyTorchModel(ModelBase):

    def __init__(
        self, 
        model: Any, 
        metadata: Mapping[str, Union[int, bool, Tuple[int, int]]]
    ) -> None: 
        self.metadata = metadata
        self.query_count = 0
        self.model = model
        self.model.eval()

    def get_preds(
        self, 
        image: ndarray
    ) -> ndarray:
        preds = self.model([image]).detach().numpy()[0]
        self.increment_query_count(1)
        return ModelBase.zip_indicies_to_preds(preds if not self.metadata['apply_softmax'] else softmax(preds))

    def get_preds_batch(
        self, 
        images: ndarray
    ) -> ndarray: 
        preds = self.model(images).detach().numpy()
        self.increment_query_count(len(images))
        return array(list(map(
            lambda x: ModelBase.zip_indicies_to_preds(x), 
            preds if not self.metadata['apply_softmax'] else map(lambda x: softmax(x), preds))))
