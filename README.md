# TIA 2022 Plates

## Clone

```sh
git clone git@github.com:caverar/TIA_2022.git
git submodule init
git submodule update
```

## Download dataset from google drive using  [gdrivedl](https://github.com/matthuisman/gdrivedl)

Is not necessary to install anything

```sh
cd data
python3 gdrivedl.py https://drive.google.com/drive/folders/1iL811t_-eqnuNwVBGeeU-HG3k6Whd3U9
unzip yolo_plate_dataset.zip
unzip yolo_plate_ocr_dataset.zip
```
