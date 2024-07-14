@echo off
:start
cls
set python_ver=36
python ./get-pip.py
pip install -r ./yolov5/requirements.txt
pip install ./yolov5/iptcinfo3-master
set /p "image_folder=Please provide path to folder which contains the subfolders for images: "
set /p "segregation_folder=Please provide path to segregation folder: "


python %cd%\\"start_detect_tag.py"  %image_folder% %segregation_folder% 