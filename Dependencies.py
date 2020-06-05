import subprocess
from sys import executable

requirements = [
    "git+https://github.com/Xilinx/brevitas.git",
    "opencv-python",
    "tensorflow",
    "tensorflow_hub>=0.6.0",
    "torch",
    "torchvision",
    "PyDrive",
    "httplib2==0.15.0",
    "google-api-python-client==1.7.11",
    "google-colab",
]

def install():
    for package in requirements:
        subprocess.check_call([executable, "-m", "pip", "install", package])
