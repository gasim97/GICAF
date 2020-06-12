from gicaf.interface.ModelBase import KerasModel
import keras
from numpy import array

def ResNet50() -> KerasModel:
    keras.backend.set_learning_phase(0)
    return KerasModel(
        model=keras.applications.resnet50.ResNet50(weights='imagenet'),
        metadata={
            'height': 224,
            'width': 224,
            'channels': 3,
            'bounds': (0.0, 255.0),
            'bgr': True,
            'classes': 1000,
            'apply softmax': False,
            'weight bits': 32,
            'activation bits': 32
        }
    )
