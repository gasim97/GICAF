from abc import ABCMeta, abstractmethod
from numpy import array, arange

class LoadDataInterface:
    __metaclass__ = ABCMeta

    @classmethod
    def version(cls): return "1.0"

    @abstractmethod
    def __init__(self, ground_truth_file_path="", img_folder_path=""): 
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
        raise NotImplementedError("Loading module __init__() function missing")

    @abstractmethod
    def get_data(self, index_ranges, height, width): 
        """
        Get images and ground truths

        Parameters
        ----------
            index_ranges : list of 2-tuples with elements of type int
                Specifies the indicies of the data to load, each tuple contains 
                (start index, end index) both inclusive
            model_metadata : dict
                Model metadata dictionary populated as specified in ModelInterface.py
        Returns
        -------
            images : numpy.ndarray
                Loaded images in the correct format, i.e. preprocessing for model input
                size and value bounds 
            ground truths : numpy.ndarray
                The ground truths of the loaded images
        """
        raise NotImplementedError("Loading module get_data() function missing")

    @abstractmethod
    def save(self, x, y, name): 
        """
        Save preprocessed input data

        Parameters
        ----------
            x : numpy.ndarray
                Images
            y: numpy.ndarray
                Ground-truths
            name : string
                Output file name
        """
        raise NotImplementedError("Loading module save function missing") 

    @abstractmethod
    def load(self, name): 
        """
        Load saved preprocessed input data

        Parameters
        ----------
            name : string
                Input file name
        Returns
        -------
            images : numpy.ndarray
                Loaded images in BGR format
            ground truths: numpy.ndarray
                The ground truths of the loaded images
        """
        raise NotImplementedError("Loading module load function missing") 

    def get_sorted_indicies_list(self, index_ranges):
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
    def close(self): 
        """
        End of session clean up
        """
        raise NotImplementedError("Loading module close() function missing")
