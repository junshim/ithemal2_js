# 1. Overview
**ithemal_transformer_two_layer** is the modified version of **ithemal_LSTM**.  
The main goal of modification is changing LSTM to transformer model
The two layer means that the instruction layer and the basic block layer are divided. 
  >   Two layer does not mean the number of encoder layers of the transformer.
  
# 2. Main files
In this chapter, describe the main files of **ithemal_transformer_two_layer**
  
## `config/`
One of main difference with **ithemal_LSTM**.  
In **ithemal_transformer_two_layer**, configuration of trainer and model is defined by using **.json** files in `config/`, not input argument like **ithemal_LSTM**.  
  
##  `run_ithemal.py` 
As the main function of **ithemal_transformer_two_layer**, create and call a trainer which do learning.  
When run `run_ithemal.py`, you can **input the configuration file of train and model as arguments** which are mentioned above `config/`.     
Examples of running `run_ithemal.py` can be showed at the **3. Training the model** chapter below.    
After the learning, this will output the trained model and its report. It will be described at the **4. Result files of training**  
  
## `train.py`
A file that defines the **trainer** which do training.  
You can change directly training configuration (batch size, learning rate, decay ... ) at `class Trainer()`.  
  
## `models.py`
A file that **defines tranformer model** for Basic Block.  
You can change directly model configuration (embedding_size, hidden_size, number of encoders, ....) at `class Ithemal()`  
  
   
# 3. Training the model 
For training the model, run the `run_ithemal.py` like below.  
`python run_ithemal.py --data {DATA_FILE} --train_cfg {TRAINER_CONFIG} --model_cfg {MODEL_CONFIG} --experiment-name {EXPERIMENT_NAME} --experiment-time {EXPERIMENT_TIME}`
* `{DATA_FILE}`: data file for training. Its 80% will be used to train.
* `{TRAINER_CONFIG}`:  configuration file of trainer. Example, `config/ithemal_train.json`
* `{MODEL_CONFIG}`: configuration file of transformer model. Example, `config/ithemal_model`: 
* `{EXPERIMENT_NAME}` and `{EXPERIMENT_TIME}`: training result are saved at `saved/{Experiment_name}/{Experiment_time}`
    
**There is an example at `run.sh`.**  
Sample Training data is saved at `trining_data/`
  
  
# 4. Result files of training
Output/result files of `run_ithemal.py` is **Trained model and its report**.  
They are saved at `saved/{Experiment_name}/{Experiment_time}`.  
* `loss_report`: (epoch, time, average loss of epoch, learning rate)  .
* `validation_result`:  (predicted, actual), (average loss), (correct-75%, total validation data).
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
