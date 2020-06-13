from typing import List, Union, Any, Mapping, Union, Tuple, Type
from abc import ABC, abstractmethod
import gicaf.Utils as utils
from tensorflow.lite.python.interpreter import Interpreter
from numpy import array, arange, flip, expand_dims, float32, ndarray
from scipy.special import softmax
from copy import deepcopy

metadata_fields = [
    'height',
    'width',
    'channels',
    'bounds',
    'bgr',
    'classes',
    'apply softmax',
    'weight bits',
    'activation bits'
]

class ModelBase(ABC):

    @classmethod
    def version(cls) -> str: return "1.0"

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

    def __init__(
        self, 
        model: Any, 
        metadata: Mapping[str, Union[int, bool, Tuple[float, float]]]
    ) -> None: 
        """
        Initialize model

        Parameters
        ----------
            model : Any
                A model of the type expected by the inheriting class
            metadata : dict
                To be stored as an instance variable 'self.metadata' and contains the 
                following fields
                    'height' : int
                        Input height
                    'width' : int
                        Input width
                    'channels' : int
                        Input number of channels
                    'bounds' : 2-tuple with elements of type float
                        Contains (min, max) of input values
                    'bgr' : bool
                        True if input is to be loaded in BGR format, and False if RGB or otherwise
                    'classes' : int
                        Number of output classes
                    'apply softmax' : bool
                        Whether or not to apply softmax to the model's output
                    'weight bits': int
                        Model weights bit width
                    'activation bits': int
                        Model activations bit width
        """
        for field in metadata_fields:
            if field not in metadata:
                raise ValueError("Model metadata dictionary is missing the '" + field + "' field")
        self.model = model
        self.metadata = metadata
        self.query_count = 0

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
            images : numpy.ndarray with elements (images) of type numpy.ndarrays
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
            images : numpy.ndarray with elements (images) of type numpy.ndarrays
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
            images : numpy.ndarray with elements (images) of type numpy.ndarrays
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
            images : numpy.ndarray with elements (images) of type numpy.ndarrays
                Images
        Returns
        -------
            [[[label, probability]]] : numpy.ndarray -> shape = (batch size, 5, 2)
                The output indicies corresponding to the highest 5 predictions
                for each image zipped with the predictions
        """
        preds = self.get_preds_batch(images)
        return flip(array(list(map(
            lambda x: x[x[:, 1].argsort()][-5:], 
            preds
        ))), 1)

    def get_top_n(
        self, 
        image: ndarray,
        n: int
    ) -> ndarray: 
        """
        Run inference and return top n ORDERED HIGHEST TO LOWEST

        Parameters
        ----------
            images : numpy.ndarray with elements (images) of type numpy.ndarrays
                Images
            n : int
                The number of top predictions to return
        Returns
        -------
            [[label, probability]]  : numpy.ndarray -> shape = (n, 2)
                The output indicies corresponding to the highest n predictions
                zipped with the predictions
        """
        if n > self.metadata['classes']:
            n = self.metadata['classes']
        preds = self.get_preds(image)
        return flip(preds[preds[:, 1].argsort()][-n:], 0) # sort predictions by probability, 
        # then extract the last n entries (highest probabilities with labels) and flip to sort 
        # by descending probability

    def get_top_n_batch(
        self, 
        images: ndarray,
        n: int
    ) -> ndarray: 
        """
        Run inference on batch and return top n ORDERED HIGHEST TO LOWEST

        Parameters
        ----------
            images : numpy.ndarray with elements (images) of type numpy.ndarrays
                Images
            n : int
                The number of top predictions to return
        Returns
        -------
            [[[label, probability]]] : numpy.ndarray -> shape = (batch size, n, 2)
                The output indicies corresponding to the highest n predictions
                for each image zipped with the predictions
        """
        if n > self.metadata['classes']:
            n = self.metadata['classes']
        preds = self.get_preds_batch(images)
        return flip(array(list(map(lambda x: x[x[:, 1].argsort()][-n:], preds))), 1)

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

    def get_preds(
        self, 
        image: ndarray
    ) -> ndarray:
        preds = self.model.predict(image)
        self.increment_query_count(1)
        return ModelBase.zip_indicies_to_preds(preds if not self.metadata['apply softmax'] else softmax(preds))

    def get_preds_batch(
        self, 
        images: ndarray
    ) -> ndarray: 
        preds = self.model.predict(images)
        self.increment_query_count(len(images))
        return array(list(map(
            lambda x: ModelBase.zip_indicies_to_preds(x), 
            preds if not self.metadata['apply softmax'] else map(lambda x: softmax(x), preds))))

class TfLiteModel(ModelBase):

    def __init__(
        self, 
        interpreter: Interpreter, 
        metadata: Mapping[str, Union[int, bool, Tuple[int, int]]]
    ) -> None: 
        interpreter.allocate_tensors()
        self.input_index = interpreter.get_input_details()[0]["index"]
        self.output_index = interpreter.get_output_details()[0]["index"]
        super(TfLiteModel, self).__init__(model=interpreter, metadata=metadata)

    def _evaluate(
        self, 
        image: ndarray
    ) -> List:
        img = expand_dims(image, axis=0).astype(float32)
        self.model.set_tensor(self.input_index, img)
        self.model.invoke()
        output = self.model.tensor(self.output_index)
        return deepcopy(output()[0])

    def get_preds(
        self, 
        image: ndarray
    ) -> ndarray:
        preds = self._evaluate(image)
        self.increment_query_count(1)
        return ModelBase.zip_indicies_to_preds(preds if not self.metadata['apply softmax'] else softmax(preds))

    def get_preds_batch(
        self, 
        images: ndarray
    ) -> ndarray: 
        self.increment_query_count(len(images))
        return array(list(map(
            lambda img: ModelBase.zip_indicies_to_preds(self._evaluate(img) if not self.metadata['apply softmax'] else softmax(self._evaluate(img))), 
            images)))

    @classmethod
    def from_saved_model(
        cls, 
        saved_model_path: str,
        model_name: str,
        bit_width: int,
        metadata: Mapping[str, Union[int, bool, Tuple[int, int]]]
    ) -> Type[ModelBase]:
        interpreter, weight_bits, activation_bits = utils.saved_model_to_tflite(
            saved_model_path=saved_model_path, 
            model_name=model_name, 
            bit_width=bit_width
        )
        metadata['weight bits'] = weight_bits
        metadata['activation bits'] = activation_bits
        return TfLiteModel(interpreter, metadata)

    @classmethod
    def from_tensorflowhub(
        cls, 
        link: str,
        model_name: str,
        bit_width: int,
        metadata: Mapping[str, Union[int, bool, Tuple[int, int]]]
    ) -> Type[ModelBase]:
        interpreter, weight_bits, activation_bits = utils.tfhub_to_tflite_converter(
            link=link, 
            model_name=model_name, 
            input_dims=[None, metadata['height'], metadata['width'], metadata['channels']],
            bit_width=bit_width
        )
        metadata['weight bits'] = weight_bits
        metadata['activation bits'] = activation_bits
        return TfLiteModel(interpreter, metadata)

class PyTorchModel(ModelBase):

    def __init__(
        self, 
        model: Any, 
        metadata: Mapping[str, Union[int, bool, Tuple[int, int]]]
    ) -> None: 
        model.eval()
        super(PyTorchModel, self).__init__(model=model, metadata=metadata)

    def get_preds(
        self, 
        image: ndarray
    ) -> ndarray:
        preds = self.model([image]).detach().numpy()[0]
        self.increment_query_count(1)
        return ModelBase.zip_indicies_to_preds(preds if not self.metadata['apply softmax'] else softmax(preds))

    def get_preds_batch(
        self, 
        images: ndarray
    ) -> ndarray: 
        preds = self.model(images).detach().numpy()
        self.increment_query_count(len(images))
        return array(list(map(
            lambda x: ModelBase.zip_indicies_to_preds(x), 
            preds if not self.metadata['apply softmax'] else map(lambda x: softmax(x), preds))))
