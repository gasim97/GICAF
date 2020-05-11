from abc import ABCMeta, abstractmethod
from numpy import array, arange

class LoadDataInterface:
    __metaclass__ = ABCMeta

    @classmethod
    def version(cls): return "1.0"

    # initialize paths
    @abstractmethod
    def __init__(self, ground_truth_file_path, img_folder_path): 
        print("Loading module __init__() function missing") 
        raise NotImplementedError

    # get images and ground truths
    @abstractmethod
    def get_data(self, index_ranges, height, width): 
        """
        index_ranges: array of tuples specifying the indicies of the data to load, each contains (start index, end index) both inclusive
        height: model input height
        width: model input width
        """
        print("Loading module get_data() function missing") 
        raise NotImplementedError
    # returns [images], [ground truths]

    # get bgr images and ground truths
    @abstractmethod
    def get_data_bgr(self, index_ranges, height, width): 
        """
        index_ranges: array of tuples specifying the indicies of the data to load, each contains (start index, end index) both inclusive
        height: model input height
        width: model input width
        """
        print("Loading module get_data_bgr() function missing") 
        raise NotImplementedError
    # returns [images], [ground truths]

    # # loads images (jpg files) and ground truths (txt file)
    # @abstractmethod
    # def load_data(self): 
    #     print("Loading module load_data function missing") 
    #     raise NotImplementedError

    # # saves preprocessed x_val and y_val binaries
    # @abstractmethod
    # def save_binary(self): 
    #     print("Loading module save_binary function missing") 
    #     raise NotImplementedError

    # # load preprocessed x_val and y_val binaries
    # @abstractmethod
    # def load_binary(self): 
    #     print("Loading module load_binary function missing") 
    #     raise NotImplementedError

    # get a list of indicies unpacked from the index ranges
    def get_sorted_indicies_list(self, index_ranges):
        indicies = [val for sublist in array(list(map(lambda x: arange(x[0], x[1] + 1), index_ranges))) for val in sublist] # unpack the index ranges to a list of indicies
        indicies = sorted(list(dict.fromkeys(indicies))) # remove duplicate indicies, incase inputed index ranges overlap, and sort
        return indicies

    # end of session clean up
    @abstractmethod
    def close(self): 
        print("Loading module close() function missing") 
        raise NotImplementedError
