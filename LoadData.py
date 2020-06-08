from typing import Optional, Tuple, Callable, List, Mapping, Union
from gicaf.interface.LoadDataBase import LoadDataBase
from numpy import array, arange, asarray, float32, ndarray
from cv2 import cvtColor, COLOR_RGB2BGR
from logging import info
from keras.preprocessing.image import load_img
from pickle import dump, load
from pathlib import Path, PosixPath
from os.path import abspath, dirname

# TODO 
# Improve preprocessing options? 
# Update load function to use less memory 

class LoadData(LoadDataBase):

    def __init__(
        self, 
        ground_truth_file_path: Optional[str] = None, 
        img_folder_path: Optional[str] = None
    ) -> None: 
        if ground_truth_file_path and img_folder_path:
            self.ground_truth_file_path = ground_truth_file_path
            self.img_folder_path = img_folder_path
            return
        parentdir = abspath('')
        self.ground_truth_file_path = parentdir + "/data/val.txt"
        self.img_folder_path = parentdir + "/data/ILSVRC2012_img_val/"

    def read_txt_file(
        self, 
        index_ranges: List[Tuple[int, int]]
    ):
        info("Reading dataset text file (file path = '" + self.ground_truth_file_path + "')...")
        sorted_indicies = LoadDataBase.get_sorted_indicies_list(index_ranges)
        txt_file = open(self.ground_truth_file_path, "r")

        data = [] # initialize, will store [[image file name, ground truth]]
        curr_index = 0 # initialize current index
        while(len(sorted_indicies) > 0):
            if sorted_indicies[0] == curr_index:
                data.append(txt_file.readline().split(" ")) # load and append data if current index is the next index in the desired index range
                sorted_indicies.pop(0) # remove the first item as the corresponding data has been loaded
            else:
                txt_file.readline() # skip data if current index is NOT the next index in the desired index range
            curr_index += 1 # increment current index

        data = list(map(lambda x: [x[0], int(x[1].strip())], data)) # strip white spaces/new lines from loaded data
        info("Test set successfully read.")
        return data

    def _get_data(self) -> Tuple[ndarray, int]:
        for i, image in enumerate(self.images_metadata):
            x = asarray(load_img(self.img_folder_path + image[0], target_size=(self.model_metadata['height'], self.model_metadata['width']), color_mode='rgb'))
            if (self.model_metadata['bgr']):
                x = asarray(cvtColor(x, COLOR_RGB2BGR))
            if (self.model_metadata['bounds'] != (0, 255)):
                x = self.preprocessing(x, self.model_metadata['bounds'])
            y = self.images_metadata[i][1]
            yield x, y

    def get_data(
        self, 
        index_ranges: List[Tuple[int, int]], 
        model_metadata: Mapping[str, Union[int, bool, Tuple[int, int]]]
    ) -> Callable[[None], Tuple[ndarray, int]]:
        self.model_metadata = model_metadata
        self.images_metadata = self.read_txt_file(index_ranges) # get image file names and ground truths
        return self._get_data

    def preprocessing(
        self, 
        image: ndarray, 
        bounds: Tuple[int, int] = (0, 1), 
        dtype=float32
    ) -> ndarray:
        info("Preprocessing image bounds.")
        divisor = 255/(bounds[1] - bounds[0])
        return array(list(map(lambda i: array(list(map(lambda j: asarray(list(map(lambda k: k/divisor + bounds[0], j)), dtype=dtype), i))), image)))

    def _save_dir(self) -> PosixPath:
        save_dir = Path(dirname(__file__) + "/tmp/saved_input_data/")
        save_dir.mkdir(exist_ok=True, parents=True)
        return save_dir

    def _save_file(
        self, 
        name: str
    ) -> PosixPath:
        save_dir = self._save_dir()
        return save_dir/(name + ".txt")

    def save(
        self, 
        data_generator: Callable[[None], Tuple[ndarray, int]], 
        name: str
    ) -> None:
        data = []
        for sample in data_generator:
            data.append(sample)
        with open(str(self._save_file(name)), "wb") as fn: 
            dump(data, fn)

    def _load(self) -> Tuple[ndarray, int]:
        for x, y in self.loaded_data:
            yield x, y

    def load(
        self, 
        name: str, 
        index_ranges: Optional[List[Tuple[int, int]]] = None
    ) -> Callable[[None], Tuple[ndarray, int]]:
        with open(str(self._save_file(name)), "rb") as fn: 
            self.loaded_data = load(fn)
        if index_ranges:
            sorted_indicies = LoadDataBase.get_sorted_indicies_list(index_ranges)
            self.loaded_data = list(map(lambda x: x[1], filter(lambda x: x[0] in sorted_indicies, enumerate(self.loaded_data))))
        return self._load

    # end of session clean up
    def close(self) -> None: 
        pass
