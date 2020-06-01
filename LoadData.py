from gicaf.interface.LoadDataInterface import LoadDataInterface
from numpy import array, arange, asarray, float32
from cv2 import cvtColor, COLOR_RGB2BGR
from logging import info
from keras.preprocessing.image import load_img
from pickle import dump, load
from pathlib import Path
from os.path import dirname

# TODO Improve preprocessing options

class LoadData(LoadDataInterface):

    def __init__(self, test_set_file_path="", img_folder_path=""):
        self.test_set_file_path = test_set_file_path
        self.img_folder_path = img_folder_path

    def read_txt_file(self, index_ranges):
        info("Reading dataset text file (file path = '" + self.test_set_file_path + "')...")
        sorted_indicies = self.get_sorted_indicies_list(index_ranges)
        txt_file = open(self.test_set_file_path, "r")

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

    def get_data(self, index_ranges, model_metadata): 
        images_metadata = self.read_txt_file(index_ranges) # get image file names and ground truths
        info("Loading images (directory path = '" + self.img_folder_path + "')...")
        x = list(map(lambda x: asarray(load_img(self.img_folder_path + x[0], target_size=(model_metadata['height'], model_metadata['width']), color_mode='rgb')), images_metadata)) # load images for file names in meta_data
        if (model_metadata['bgr']):
            x = asarray(list(map(lambda x: cvtColor(x, COLOR_RGB2BGR), x)))
        y = asarray(list(map(lambda x: x[1], images_metadata))) # unpack ground truths from meta_data
        info("Images successfully loaded.")
        if (model_metadata['bounds'] != (0, 255)):
            x = self.preprocessing(x, model_metadata['bounds'])
        return x, y

    def preprocessing(self, x, bounds=(0, 1), dtype=float32):
        info("Preprocessing image bounds.")
        divisor = 255/(bounds[1] - bounds[0])
        return array(list(map(lambda img: array(list(map(lambda i: array(list(map(lambda j: asarray(list(map(lambda k: k/divisor + self.bounds[0], j)), dtype=dtype), i))), img))), x)))

    def _save_dir(self):
        save_dir = Path(dirname(__file__) + "/tmp/saved_input_data/")
        save_dir.mkdir(exist_ok=True, parents=True)
        return save_dir

    def _save_file(self, name):
        save_dir = self._save_dir()
        return save_dir/(name + ".txt")

    def save(self, x, y, name):
        data = list(zip(x, y))
        with open(str(self._save_file(name)), "wb") as fn: 
            dump(data, fn)

    def load(self, name):
        with open(str(self._save_file(name)), "rb") as fn: 
            data = load(fn)
        return array(list(map(lambda x: x[0], data))), array(list(map(lambda y: y[1], data)))

    # end of session clean up
    def close(self): 
        pass
