# Automatic Number Plate Recognition (ANPR)

This repo contains the development of the project for the course of Artificial Intelligence Techniques at the National University of Colombia. This project consists of the development of an Automatic Number Plate Recognition (ANPR) system using image processing techniques and machine learning algorithms.

The members of the team are:

* [Camilo Andres Vera Ruiz](https://github.com/caverar)
* [Cristian Yesid Chitiva Vela](https://github.com/cychitivav)
* [Maria Alejandra Arias Frontanilla](https://github.com/ariasAleia)

## Dependencies

```bash
pip install numpy
pip install matplotlib
pip install opencv-contrib-python
pip install pandas
pip install tensorflow
```

## Clone and download datasets

Clone with submodules and download from google drive some datasets automatically using [gdrivedl](https://github.com/matthuisman/gdrivedl) and [termcolor](https://pypi.org/project/termcolor/).

```bash
git clone --recurse-submodules https://github.com/caverar/TIA_2022.git
cd TIA_2022
python lib/datadl.py
```

The datasets were taken from here:

* [license-plate-dataset](https://github.com/RobertLucian/license-plate-dataset)
* [Car_License_Plate_Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)
* [Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano](https://github.com/winter2897/Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano)

## License

[MIT License](LICENSE)
