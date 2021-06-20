from typing import Optional, Union, List, Callable, Tuple
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.lite as lite
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.lite.python.interpreter import Interpreter
from numpy import array, ndarray
from logging import info
from pathlib import Path, PosixPath
from os import path, walk, remove
from zipfile import ZipFile, ZIP_DEFLATED
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pydrive.files import GoogleDriveFile
from google.colab import auth
from oauth2client.client import GoogleCredentials

# TensorFlow Lite and hub helper functions

def create_dummy_sample(dims: List[Union[Optional[int]]] = [None, 224, 224, 3]) -> ndarray:
    return array(list(map(lambda i: array(list(map(lambda j: array(list(map(lambda k: array(list(map(lambda z: 0.0, range(dims[3])))), range(dims[2])))), range(dims[1])))), range(dims[0] if dims[0] != None else 1))))

def _tfhub_model_dir(model_name: str, create_dir: bool = True) -> PosixPath:
    tfhub_models_dir = Path(path.dirname(__file__) + "/tmp/tfhub_models/" + model_name + "/")
    if (create_dir):
        tfhub_models_dir.mkdir(exist_ok=True, parents=True)
    return tfhub_models_dir

def _tflite_model_dir(model_name: str, create_dir: bool = True) -> PosixPath:
    tflite_models_dir = Path(path.dirname(__file__) + "/tmp/tflite_models/" + model_name + "/")
    if (create_dir):
        tflite_models_dir.mkdir(exist_ok=True, parents=True)
    return tflite_models_dir

def _tflite_model_file(model_name: str, bit_width: int = 8) -> PosixPath:
    tflite_models_dir = _tflite_model_dir(model_name)
    return tflite_models_dir/("model_" + str(bit_width) + "bit.tflite")

def saved_model_to_tflite(
    saved_model_path: str, 
    model_name: str,
    representative_dataset: Callable[[None], ndarray],
    bit_width: int = 8
) -> Tuple[Interpreter, int, int]:
    info("Converting saved model to tfLite model")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    if (bit_width != 32):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
    if (bit_width == 8):
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    elif (bit_width == 16):
        converter.target_spec.supported_types = [tf.float16]
    elif (bit_width != 32):
        raise ValueError('Expected a bit width 8, 16 or 32. Got a bit width of {}.'.format(bit_width))
    weight_bits = bit_width
    activation_bits = bit_width
    tflite_model = converter.convert()
    tflite_model_file = _tflite_model_file(model_name, bit_width)
    info("Saving tfLite model to " + str(tflite_model_file))
    tflite_model_file.write_bytes(tflite_model)
    interpreter = lite.Interpreter(model_path=str(tflite_model_file))
    return interpreter, weight_bits, activation_bits

def get_tfhub_model(link: str) -> Sequential:
    info("Getting tfHub model")
    return tf.keras.Sequential([hub.KerasLayer(link)])

def save_tfhub_model(model, model_name: str) -> PosixPath:
    tfhub_models_dir = _tfhub_model_dir(model_name)
    info("Saving tfHub model to " + str(tfhub_models_dir))
    model.save(tfhub_models_dir, include_optimizer=False)
    return tfhub_models_dir

def tfhub_to_tflite_converter(
    link: str, 
    model_name: str, 
    representative_dataset: Callable[[None], ndarray],
    input_dims: List[Union[Optional[int]]] = [None, 224, 224, 3], 
    bit_width: int = 8
) -> Tuple[Interpreter, int, int]:
    tflite_model_file = _tflite_model_file(model_name, bit_width)
    if (tflite_model_file.exists()):
        info("Using saved tflite model at " + str(tflite_model_file))
        interpreter = lite.Interpreter(model_path=str(tflite_model_file))
        weight_bits = bit_width
        activation_bits = bit_width
        return interpreter, weight_bits, activation_bits
    tfhub_models_dir = _tfhub_model_dir(model_name, create_dir=False)
    if (tfhub_models_dir.is_dir()):
        info("Using saved tfhub model at " + str(tfhub_models_dir))
    else:
        model = get_tfhub_model(link)
        info("Compiling tfHub model")
        model.predict(create_dummy_sample(input_dims))
        model.compile()
        tfhub_models_dir = save_tfhub_model(model, model_name)
    interpreter, weight_bits, activation_bits = saved_model_to_tflite(str(tfhub_models_dir), model_name, representative_dataset, bit_width)
    return interpreter, weight_bits, activation_bits

# Google Drive helper functions

def _remove_file(file_name: str = 'tmp.zip') -> None:
    remove(Path(path.dirname(__file__) + "/" + file_name))

def _zip_dir(dir_name: str = "tmp") -> None:
    with ZipFile('gicaf/' + dir_name + '.zip', 'w', ZIP_DEFLATED) as zip_file:
        for folder_name, _, file_names in walk('gicaf/' + dir_name):
            for file_name in file_names:
                file_path = path.join(folder_name, file_name)
                info("Compressing " + str(file_path))
                zip_file.write(file_path)

def _unzip_file(file_name: str = 'tmp') -> None:   
    with ZipFile('gicaf/' + file_name + '.zip', 'r') as zip_file:
        zip_file.extractall()

def _get_gdrive_drive() -> GoogleDrive:
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    return drive

def _get_gdrive_file_metadata(file_name: str) -> GoogleDriveFile:
    drive = _get_gdrive_drive()
    file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
    for file_ in file_list:
        if file_['title'] == file_name:
            return file_
    raise NameError("Failed to find '" + file_name + "' on Google Drive, enure that the tmp folder has been saved before and the correct file name is being used")

def _save_tmp_to_new_gdrive(gdrive_file_name: str = 'gicaf_tmp') -> None:
    gdrive_file_name = gdrive_file_name + ".zip"
    drive = _get_gdrive_drive()
    _zip_dir()
    upload = drive.CreateFile({'title': gdrive_file_name})
    upload.SetContentFile('gicaf/tmp.zip')
    upload.Upload()
    _remove_file()
    info("Saved compressed GICAF tmp folder to Google Drive as '" + gdrive_file_name + "'")

def save_tmp_to_gdrive(gdrive_file_name: str = 'gicaf_tmp') -> None:  
    upload = _get_gdrive_file_metadata(gdrive_file_name + ".zip")
    if (upload == None):
        _save_tmp_to_new_gdrive(gdrive_file_name)
        return
    _zip_dir()
    upload.SetContentFile('gicaf/tmp.zip')
    upload.Upload()
    _remove_file()
    info("Updated compressed GICAF tmp folder '" + gdrive_file_name + ".zip' on Google Drive")

def load_tmp_from_gdrive(gdrive_file_name: str = 'gicaf_tmp') -> None:
    gdrive_file_name = gdrive_file_name + ".zip"
    drive = _get_gdrive_drive()
    download = drive.CreateFile({'id': _get_gdrive_file_metadata(gdrive_file_name)['id']})
    download.GetContentFile('gicaf/tmp.zip')
    _unzip_file()
    _remove_file()
    info("Loaded GICAF tmp folder from '" + gdrive_file_name + "' on Google Drive")
