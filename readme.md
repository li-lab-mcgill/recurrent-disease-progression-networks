# Installation

Install numpy, sklearn, tensorflow, pandas, matplotlib, tqdm, imblearn via pip
```sh
pip install numpy
pip install scikit-learn
pip install tensorflow
pip install pandas
pip install matplotlib
pip install tqdm
pip install imblearn
```
# Preprocessing

The tool takes the raw data in csv format, named patient_data.csv. Designate the path to patient_data.csv as the following:
```sh
python preprocess.py <path to the folder containing patient_data.csv>
```
Processed data will be saved in the same folder.

# Training
Execute train.py to train different models by specifying the following arguments:
```sh
python train.py <path to the folder containing processed data> <mode> <loss> <architecture> <number of layers>
```
- Path: Enter the relative path to the folder containing processed data
- Mode: Speficy the model type. Available options are: lr, svm, standard, dhtm, and dhtmc
- Loss: Specify the loss type. Available options are: bce, focal_loss, and balanced_bce
- Architecture: Specify the basic RNN blocks applied in the model. Available options are: gru, lstm, rnn.
- Number of layers: Specify number of basic RNN blocks in the model. 

For the two modes, lr and svm, enter none for both loss and architecture. Enter any number for number of layers.
Model and model history will be saved in two separate folders in the folder containing the data. For lr and svm, the models are in .joblib format, while other models are in .h5 format.

# Trajectory Prediction
Execute trajectory.py to evaluate heart failure trajectory prediction by specifying the following arguments:
```sh
python trajectory.py <path to the folder containing processed data> <mode> <path to the model>
```
- Path: Enter the relative path to the folder containing processed data
- Mode: Speficy the model type. Available options are: standard, dhtm, and dhtmc
For lr and svm modes, enter none for both loss and architecture and any number for number of layers.
- Model path: Enter the relative path to the model

The generated plots are saved in the plots under the current working directory.

# Evaluation
Execute eval.py to evaluate the model by specifying the following arguments:
```sh
python eval.py <path to the folder containing processed data> <mode> <path to the model>
```
The inputs are similar to trajectory.py, but the mode available are: lr, svm, standard, dhtm, and dhtmc. The generated plots are saved in the plots under the current working directory.
