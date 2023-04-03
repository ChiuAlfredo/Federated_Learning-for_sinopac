# create conda env  
conda create -n ./env python=3.10.4

# install tensorflow_dev for mac
conda install -c apple tensorflow-deps

# install tendowflow-mac
python -m pip install tensorflow-macos

# install tensorflow-metal
python -m pip install tensorflow-metal

# pip install 
pip install -r requirements.txt