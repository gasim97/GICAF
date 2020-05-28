from gicaf.interface.ModelInterface import KerasModel
import keras
from numpy import array

class ResNet50(KerasModel):

    def __init__(self): 
        keras.backend.set_learning_phase(0)
        super(ResNet50, self).__init__(kmodel=keras.applications.resnet50.ResNet50(weights='imagenet'),
                                        metadata={'height': 224, 
                                                    'width': 224, 
                                                    'channels': 3, 
                                                    'bounds': (0, 255),
                                                    'bgr': True, 
                                                    'classes': 1000, 
                                                    'bounds': (0, 255),
                                                    'weight_bits': 32,
                                                    'activation_bits': 32})
    

