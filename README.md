# TIA 2022 Plates

## Clone

```sh
git clone git@github.com:caverar/TIA_2022.git
cd TIA_2022
git submodule init
git submodule update
```

## Download dataset from google drive using  [gdrivedl](https://github.com/matthuisman/gdrivedl)

Is not necessary to install anything

```sh
cd data
mkdir Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano
mkdir Car_License_Plate_Detection
python3 gdrivedl.py https://drive.google.com/drive/folders/1iL811t_-eqnuNwVBGeeU-HG3k6Whd3U9
mv yolo_plate_dataset.zip Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano
mv yolo_plate_ocr_dataset.zip Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano
cd Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano
unzip yolo_plate_dataset.zip
unzip yolo_plate_ocr_dataset.zip
cd ..
mv archive.zip Car_License_Plate_Detection
cd Car_License_Plate_Detection
unzip archive.zip
```
