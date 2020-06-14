import setuptools


install_requires = [
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

setuptools.setup(
    name="GICAF",
    version="0.0.1",
    description="Framework and toolkit for running black-box adversarial attack experiments",  # noqa: E501
    long_description='',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
    ],
    author="Gasim Gasim",
    author_email="gasim97@gmail.com",
    url="https://github.com/gasim97/gicaf",
    license="MIT",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    python_requires='>=3.7',
)