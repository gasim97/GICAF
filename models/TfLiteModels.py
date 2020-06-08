from gicaf.interface.ModelBase import TfLiteModel
import gicaf.Utils as utils

class ResNet50(TfLiteModel):

    # initialize
    def __init__(
        self, 
        bit_width: int = 32
    ) -> None: 
        # sets up local ResNet50 model, to use for local testing
        interpreter, weight_bits, activation_bits = utils.tfhub_to_tflite_converter("https://tfhub.dev/tensorflow/resnet_50/classification/1", "resnet_50", bit_width=bit_width)
        super(ResNet50, self).__init__(interpreter=interpreter, 
                                        metadata={'height': 224, 
                                                    'width': 224, 
                                                    'channels': 3,
                                                    'bounds': (0, 1), 
                                                    'bgr': False,
                                                    'classes': 1000, 
                                                    'apply_softmax': False,
                                                    'weight_bits': weight_bits,
                                                    'activation_bits': activation_bits})


class EfficientNetB0(TfLiteModel):

    # initialize
    def __init__(
        self, 
        bit_width: int = 32
    ) -> None: 
        # sets up local EfficientNet-B0 model, to use for local testing
        interpreter, weight_bits, activation_bits = utils.tfhub_to_tflite_converter("https://tfhub.dev/tensorflow/efficientnet/b0/classification/1", "efficientnet_b0", bit_width=bit_width)
        super(EfficientNetB0, self).__init__(interpreter=interpreter, 
                                        metadata={'height': 224, 
                                                    'width': 224, 
                                                    'channels': 3,
                                                    'bounds': (0, 1), 
                                                    'bgr': False,
                                                    'classes': 1000, 
                                                    'apply_softmax': True,
                                                    'weight_bits': weight_bits,
                                                    'activation_bits': activation_bits})

class EfficientNetB7(TfLiteModel):

    # initialize
    def __init__(
        self, 
        bit_width: int = 32
    ) -> None: 
        # sets up local EfficientNet-B7 model, to use for local testing
        interpreter, weight_bits, activation_bits = utils.tfhub_to_tflite_converter("https://tfhub.dev/google/efficientnet/b7/classification/1", "efficientnet_b7", bit_width=bit_width)
        super(EfficientNetB7, self).__init__(interpreter=interpreter, 
                                        metadata={'height': 224, 
                                                    'width': 224, 
                                                    'channels': 3,
                                                    'bounds': (0, 1), 
                                                    'bgr': False,
                                                    'classes': 1000, 
                                                    'apply_softmax': True,
                                                    'weight_bits': weight_bits,
                                                    'activation_bits': activation_bits})
