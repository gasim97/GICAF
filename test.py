from Logger import Logger
from LoadData import LoadData
from os.path import abspath
from numpy import shape
import logging
logging.basicConfig(level=logging.INFO)

parentdir = abspath('')
print(parentdir)
test_set_file_path = parentdir + "/data/val.txt"
img_folder_path = parentdir + "/data/ILSVRC2012_img_val/"

def testLoadData():
    img_shape = (224, 224, 3)
    load_data = LoadData(test_set_file_path, img_folder_path)
    data = load_data.get_data([(0,0)], img_shape[0], img_shape[1])
    data_bgr = load_data.get_data_bgr([(0,0)], 224, 224)

    assert shape(data[0][0]) == img_shape

    print(shape(data))
    print(data[0][0][0][0])
    print(data_bgr[0][0][0][0])

def testLogger():
    logger = Logger()
    logger.nl(["l1col1", "l1col2"])
    logger.append({
        "l1col1": 1,
        "l1col2": 0
    })
    logger.append([2, 0])
    logger.append([3, 0])
    logger.nl(["l2col1", "l2col2"])
    logger.append([1, 0])
    logger.append([2, 0])
    logger.append([3, 0])

# testLoadData()
# testLogger()