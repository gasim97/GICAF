from gicaf.interface.ModelInterface import TfLiteModel
import gicaf.Utils as utils

class ResNet50(TfLiteModel):

    # initialize
    def __init__(self, bit_width=32): 
        # sets up local ResNet50 model, to use for local testing
        tflite_model_file_path = utils.tfhub_to_tflite_converter("https://tfhub.dev/tensorflow/resnet_50/classification/1", "resnet_50", bit_width=bit_width)
        super(ResNet50, self).__init__(tflite_model_file_path=tflite_model_file_path)
    
    # get model metadata
    def metadata(self): 
        return {'height': 224, 'width': 224, 'channels': 3, 'bgr': False}
    # returns input height, input width, input channels, (True if BGR else False) in a dictionary

