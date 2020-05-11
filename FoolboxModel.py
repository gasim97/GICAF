from Interfaces.ModelInterface import FoolboxKerasModelInterface
import keras
from numpy import array

class ResNet50(FoolboxKerasModelInterface):

    # initialize
    def __init__(self): 
        # sets up local ResNet50 model, to use for local testing
        keras.backend.set_learning_phase(0)
        super(ResNet50, self).__init__(kmodel=keras.applications.resnet50.ResNet50(weights='imagenet'),
                                        preprocessing=(array([104, 116, 123]), 1))
    
    # get model metadata
    def metadata(self): 
        return {'height': 224, 'width': 224, 'channels': 3, 'bgr': True}
    # returns input height, input width, input channels, (True if BGR else False) in a dictionary

