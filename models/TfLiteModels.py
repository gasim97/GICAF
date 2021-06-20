from typing import List, Tuple, Callable
from numpy import ndarray
from gicaf.interface.ModelBase import TfLiteModel
from gicaf.interface.LoadDataBase import LoadDataBase

def ResNet50(loadData: LoadDataBase, bit_width: int = 32, rep_data_index_ranges: List[Tuple[int, int]] = [(0, 50)]) -> TfLiteModel: 
    return TfLiteModel.from_tensorflowhub(
        link="https://tfhub.dev/tensorflow/resnet_50/classification/1",
        model_name="resnet_50",
        bit_width=bit_width,
        loadData=loadData,
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

def EfficientNetB0(loadData: LoadDataBase, bit_width: int = 32, rep_data_index_ranges: List[Tuple[int, int]] = [(0, 50)]) -> TfLiteModel: 
    return TfLiteModel.from_tensorflowhub(
        link="https://tfhub.dev/tensorflow/efficientnet/b0/classification/1",
        model_name="efficientnet_b0",
        bit_width=bit_width,
        loadData=loadData,
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

def EfficientNetB7(loadData: LoadDataBase, bit_width: int = 32, rep_data_index_ranges: List[Tuple[int, int]] = [(0, 50)]) -> TfLiteModel: 
    return TfLiteModel.from_tensorflowhub(
        link="https://tfhub.dev/google/efficientnet/b7/classification/1",
        model_name="efficientnet_b7",
        bit_width=bit_width,
        loadData=loadData,
        metadata={
            'height': 600, 
            'width': 600, 
            'channels': 3,
            'bounds': (0.0, 1.0), 
            'bgr': False,
            'classes': 1000, 
            'apply softmax': True,
        }
    )
