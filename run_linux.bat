#!/bin/bash

clear
python_ver=36
python ./get-pip.py
pip install -r ./yolov5/requirements.txt
pip install ./yolov5/iptcinfo3-master
read -p "Please provide path to folder which contains the subfolders for images: " image_folder
read -p "Please provide path to segregation folder: " segregation_folder

python "$(pwd)/start_detect_tag.py" "$image_folder" "$segregation_folder"
