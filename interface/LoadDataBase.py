from typing import Callable, List, Tuple, Optional, Type, Mapping, Union
from abc import ABC, abstractmethod
from numpy import array, arange, ndarray

class LoadDataBase(ABC):

    @classmethod
    def version(cls): return "1.0"

    @classmethod
    def get_sorted_indicies_list(
        cls, 
        index_ranges: List[Tuple[int, int]]
    ) -> List[int]:
        """
        Get a list of indicies unpacked from the index ranges

        Parameters
        ----------
            index_ranges : list of 2-tuples with elements of type int
                List of tuples each containing (start index, end index)
        Returns
        -------
            indicies : list
                List of indicies unpacked from index_ranges
        Example
        -------
            For input index_ranges = [(1, 3), (9, 11), (5, 7), (6, 7)]
            returns indicies = [1, 2, 3, 5, 6, 7, 9, 10, 11]
        """
        indicies = [val for sublist in array(list(map(lambda x: arange(x[0], x[1] + 1), index_ranges))) for val in sublist] # unpack the index ranges to a list of indicies
        indicies = sorted(list(dict.fromkeys(indicies))) # remove duplicate indicies, incase inputed index ranges overlap, and sort
        return indicies

    @abstractmethod
    def __init__(
        self, 
        ground_truth_file_path: Optional[str] = None, 
        img_folder_path: Optional[str] = None
    ) -> None: 
        """
        Initialize data loader and data paths

        Parameters
        ----------
            ground_truth_file_path : string
                The absolute path to a file containing the file names of the images
                to be loaded and the associated ground-truths. The file should be a text
                file. The image file names and the ground-truths should be separated by 
                a space and listed line by line, as below
                File structure:
                    image_file_name ground_truth
                    image_file_name ground_truth
                    ...
            img_folder_path: sting
                The absolute path to the folder/directory containing the images to be
                loaded with file names as in the ground truths file
        """
        ...

    @abstractmethod
    def get_data(
        self, 
        index_ranges: List[Tuple[int, int]], 
        model_metadata: Mapping[str, Union[int, bool, Tuple[int, int]]]
    ) -> Callable[[None], Tuple[ndarray, int]]: 
        """
        Get images and ground truths

        Parameters
        ----------
            index_ranges : list of 2-tuples with elements of type int
                Specifies the indicies of the data to load, each tuple contains 
                (start index, end index) both inclusive
            model_metadata : dict
                Model metadata dictionary populated as specified in ModelBase.py
        Returns
        -------
            data_generator : generator function
                Yields
                ------
                    image : numpy.ndarray
                        Loaded image in the correct format, i.e. preprocessing for model input
                        size and value bounds 
                    ground truth : int
                        The ground truth of the loaded image
        """
        ...

    @abstractmethod
    def save(
        self, 
        data_generator: Callable[[None], Tuple[ndarray, int]], 
        name: str
    ) -> None: 
        """
        Save preprocessed input data

        Parameters
        ----------
            data_generator : generator function
                Yields
                ------
                    x : numpy.ndarray
                        Image
                    y : int
                        Ground-truth of x
            name : string
                Output file name
        """
        ...

    @abstractmethod
    def load(
        self, 
        name: str, 
        index_ranges: Optional[List[Tuple[int, int]]] = None
    ) -> Callable[[None], Tuple[ndarray, int]]: 
        """
        Load saved preprocessed input data

        Parameters
        ----------
            name : string
                Input file name
            index_ranges : list of 2-tuples with elements of type int
                Specifies the indicies of the data to load, each tuple contains 
                (start index, end index) both inclusive. Default is None and
                corresponds to loading all samples
        Returns
        -------
            data_generator : generator function
                Yields
                ------
                    image : numpy.ndarray
                        Loaded images in BGR format
                    ground truth : int
                        The ground truth of the loaded image
        """
        ...

    @abstractmethod
    def close(self): 
        """
        End of session clean up
        """
        ...
