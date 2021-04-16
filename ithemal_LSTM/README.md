# 1. Overview
**ithemal_LSTM** is the modified version of [**original ithemal**](https://github.com/ithemal/Ithemal) source code.  
The main goal of modification is running ithemal code at GPU. (The original version is optimized to train with CPU.)  

# 2. Main files
In this chapter, describe the main files of **ithemal_LSTM**  
##  `run_ithemal.py` 
As the main function of  **ithemal_LSTM**, create and call a trainer that do learning.  
When run `run_ithemal.py`, you **can input the configuration of train and model as arguments**. Otherwise, use the default value.  
Configuration list and default values ​​can be checked at the main function in `run_ithemal.py`.  
Examples of running `run_ithemal.py` can be showed at the **3. Training the model** chapter below.    
After the learning, this will output the trained model and its report. It will be described at the **4. Result files of training**.  


## `training.py`
A file that defines the **trainer** which do training.  
You can change directly training configuration (batch size, learning rate, decay ... ) at `class Train()`.  

## `models/graph_models.py`
A file that defines several **models** for train.  
There are tree models (RNN, LSTM, GRU), and **default model is LSTM**.  
You can change directly model configuration (embedding_size, hidden_size, ....) at `class RNN()` 
  
  
# 3. Training the model 
For training the model, run the `run_ithemal.py` like below.  
`python run_ithemal.py --data {DATA_FILE} --use-rnn train --experiment-name {EXPERIMENT_NAME} --experiment-time {EXPERIMENT_TIME}`  
* `{DATA_FILE}`: data file for training. Its 80% will be used to train and 20% will be used to validation
* `{EXPERIMENT_NAME}` and `{EXPERIMENT_TIME}`: Specify the location where the result files will be saved. They are saved into `saved/{Experiment_name}/{Experiment_time}`

**There is an example at `run.sh`.** 
Sample Training data is saved at `trining_data/`
  
  
# 4. Result files of training
Output/result files of `run_ithemal.py` is **Trained model and its report**.  
They are saved at `saved/{Experiment_name}/{Experiment_time}`.  
* `loss_report`: (epoch, time, average loss of epoch, learning rate)  .
* `validation_result`:  (predicted, actual), (average loss), (correct-75%, total validataion data).
* `trained.mdl`: model data.
* `predictor.dump`: model architecture of model.
* `checkpoints`: trained model per epoch.

# 5. Prediction
For doing prediction by using trained model, run `prediction.py` like below.  
`python prediction.py --model {MODEL_ARCHITECTURE_FILE} --model-data {MODEL_DATA_FILE} --data {DATA_FILE}`  
	* `{MODEL_ARCHITECTURE_FILE}`: Model architecture to use. It is **predictor.dump** file at  **4. Result files of training**
	* `{MODEL_DATA_FILE}`: Model data to use. It is **trained.mdl** file at **4. Result files of training**
	* `{DATA_FILE}`: data file to predict
> Currently, outputs the accuracy of the training model for the input data in data file. 

**There is an example at `prediction.sh`.**  

