from gicaf.interface.ModelBase import KerasModel
import keras
from numpy import array

class ResNet50(KerasModel):

    def __init__(self) -> None: 
        keras.backend.set_learning_phase(0)
        super(ResNet50, self).__init__(kmodel=keras.applications.resnet50.ResNet50(weights='imagenet'),
                                        metadata={'height': 224, 
                                                    'width': 224, 
                                                    'channels': 3, 
                                                    'bounds': (0, 255),
                                                    'bgr': True, 
                                                    'classes': 1000, 
                                                    'apply_softmax': False,
                                                    'weight_bits': 32,
                                                    'activation_bits': 32})
