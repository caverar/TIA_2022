# Automatic Number Plate Recognition (ANPR)

This repository contains the development of a project for the course of Artificial Intelligence Techniques at Universidad Nacional de Colombia. This project consists of the development of an Automatic Number Plate Recognition (ANPR) system using image processing techniques and machine learning algorithms.

The team members are:

* [Camilo Andres Vera Ruiz](https://github.com/caverar)
* [Cristian Yesid Chitiva Vela](https://github.com/cychitivav)
* [Maria Alejandra Arias Frontanilla](https://github.com/ariasAleia)

## Dependencies

### Python3.10

Installation process in ubuntu:

```bash
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10
```

### Libraries

if using a default os python version prior to 3.10 install everything with `pip3.10` and run scripts with `python3.10`.

```bash
pip install termcolor
pip install numpy
pip install matplotlib
pip install opencv-contrib-python
pip install pandas

pip install tensorflow

# If using WSL install this TensorFlow version instead, uninstall the last one if already installed 
pip install tensorflow-cpu
pip install tensorflow-directml-plugin
```

## Clone and download datasets

Clone with submodules and download from google drive some datasets automatically using [gdrivedl](https://github.com/matthuisman/gdrivedl) and [termcolor](https://pypi.org/project/termcolor/).

```bash
git clone --recurse-submodules https://github.com/caverar/TIA_2022.git
cd TIA_2022
python3 lib/datadl.py
```

The datasets were taken from here:

* [license-plate-dataset](https://github.com/RobertLucian/license-plate-dataset)
* [Car_License_Plate_Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)
* [Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano](https://github.com/winter2897/Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano)
* [Cars Dataset](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)

## Generate normalized images and annotations

Inside of the repository directory execute the next line. Use `python3.10` instead of `python3` if using ppa installation

```bash
python3 lib/data_normalizer.py
```

## License

[MIT License](LICENSE)
