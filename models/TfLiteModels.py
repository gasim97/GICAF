from gicaf.interface.ModelBase import TfLiteModel

def ResNet50(bit_width: int = 32) -> TfLiteModel: 
    return TfLiteModel.from_tensorflowhub(
        link="https://tfhub.dev/tensorflow/resnet_50/classification/1",
        model_name="resnet_50",
        bit_width=bit_width,
        metadata={
            'height': 224, 
            'width': 224, 
            'channels': 3,
            'bounds': (0.0, 1.0), 
            'bgr': False,
            'classes': 1000, 
            'apply softmax': False,
        }
    )

def EfficientNetB0(bit_width: int = 32) -> TfLiteModel: 
    return TfLiteModel.from_tensorflowhub(
        link="https://tfhub.dev/tensorflow/efficientnet/b0/classification/1",
        model_name="efficientnet_b0",
        bit_width=bit_width,
        metadata={
            'height': 224, 
            'width': 224, 
            'channels': 3,
            'bounds': (0.0, 1.0), 
            'bgr': False,
            'classes': 1000, 
            'apply softmax': True,
        }
    )

def EfficientNetB7(bit_width: int = 32) -> TfLiteModel: 
    return TfLiteModel.from_tensorflowhub(
        link="https://tfhub.dev/google/efficientnet/b7/classification/1",
        model_name="efficientnet_b7",
        bit_width=bit_width,
        metadata={
            'height': 224, 
            'width': 224, 
            'channels': 3,
            'bounds': (0.0, 1.0), 
            'bgr': False,
            'classes': 1000, 
            'apply softmax': True,
        }
    )
