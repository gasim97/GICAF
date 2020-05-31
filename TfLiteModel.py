from gicaf.interface.ModelInterface import TfLiteModel
import gicaf.Utils as utils

class ResNet50(TfLiteModel):

    # initialize
    def __init__(self, bit_width=32): 
        # sets up local ResNet50 model, to use for local testing
        interpreter, weight_bits, activation_bits = utils.tfhub_to_tflite_converter("https://tfhub.dev/tensorflow/resnet_50/classification/1", "resnet_50", bit_width=bit_width)
        super(ResNet50, self).__init__(interpreter=interpreter, 
                                        metadata={'height': 224, 
                                                    'width': 224, 
                                                    'channels': 3,
                                                    'bounds': (0, 1), 
                                                    'bgr': False,
                                                    'classes': 1000, 
                                                    'weight_bits': weight_bits,
                                                    'activation_bits': activation_bits})


class EfficientNetB0(TfLiteModel):

    # initialize
    def __init__(self, bit_width=32): 
        # sets up local ResNet50 model, to use for local testing
        interpreter, weight_bits, activation_bits = utils.tfhub_to_tflite_converter("https://tfhub.dev/tensorflow/efficientnet/b0/classification/1", "efficientnet_b0", bit_width=bit_width)
        super(EfficientNetB0, self).__init__(interpreter=interpreter, 
                                        metadata={'height': 224, 
                                                    'width': 224, 
                                                    'channels': 3,
                                                    'bounds': (0, 1), 
                                                    'bgr': False,
                                                    'classes': 1000, 
                                                    'weight_bits': weight_bits,
                                                    'activation_bits': activation_bits})
    

