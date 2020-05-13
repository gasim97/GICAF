import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
from numpy import array
from logging import info

def create_dummy_sample(dims=[None, 224, 224, 3]):
  return array(list(map(lambda i: array(list(map(lambda j: array(list(map(lambda k: array(list(map(lambda z: 0.0, range(dims[3])))), range(dims[2])))), range(dims[1])))), range(dims[0] if dims[0] != None else 1))))

def saved_model_to_tflite(saved_model_path, model_name, bit_width=8):
    info("Converting saved model to tfLite model")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    if (bit_width != 32):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if (bit_width == 16):
        converter.target_spec.supported_types = [tf.float16]
    elif (bit_width != 8 and bit_width != 32):
        print("Got a bit_width of " + bit_width + ". Expected 8, 16 or 32.")
        raise ValueError
    tflite_model = converter.convert()
    info("Saving tfLite model")
    tflite_models_dir = Path("tmp/tflite_models/" + model_name + "/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_file = tflite_models_dir/"model" + str(bit_width) + ".tflite"
    tflite_model_file.write_bytes(tflite_model)
    return tflite_model_file

def get_tfhub_model(link):
    info("Getting tfHub model")
    return tf.keras.Sequential([hub.KerasLayer(link)])

def save_tfhub_model(model, model_name):
    info("Saving tfHub model")
    tfhub_models_dir = Path("tmp/tfhub_models/" + model_name + "/")
    tfhub_models_dir.mkdir(exist_ok=True, parents=True)
    model.save(tfhub_models_dir, include_optimizer=False)
    return tfhub_models_dir

def tfhub_to_tflite_converter(link, model_name, input_dims=[None, 224, 224, 3], bit_width=8):
    model = get_tfhub_model(link)
    info("Compiling tfHub model")
    model.predict(create_dummy_sample(input_dims))
    model.compile()
    tfhub_models_dir = save_tfhub_model(model, model_name)
    tflite_model_file = saved_model_to_tflite(tfhub_models_dir, model_name, bit_width=8)
    return tflite_model_file