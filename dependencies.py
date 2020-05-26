import subprocess
from sys import executable

def install():
    subprocess.check_call([executable, "-m", "pip", "install", "git+https://github.com/Xilinx/brevitas.git"])
    subprocess.check_call([executable, "-m", "pip", "install", "git+https://github.com/alvarorobledo/foolbox.git"])
    subprocess.check_call([executable, "-m", "pip", "install", "opencv-python"])
    subprocess.check_call([executable, "-m", "pip", "install", "tensorflow"])
    subprocess.check_call([executable, "-m", "pip", "install", "tensorflow_hub>=0.6.0"])
    subprocess.check_call([executable, "-m", "pip", "install", "torch", "torchvision"])
    subprocess.check_call([executable, "-m", "pip", "install", "PyDrive"])
    subprocess.check_call([executable, "-m", "pip", "install", "httplib2==0.15.0"])
    subprocess.check_call([executable, "-m", "pip", "install", "google-api-python-client==1.7.11"])
    subprocess.check_call([executable, "-m", "pip", "install", "google-colab"])