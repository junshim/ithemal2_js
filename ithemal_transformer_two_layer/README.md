# Training
To train model, use `run.sh`
Data file for training in the training_data/ 

## run.sh
`python run_ithemal.py --data {DATA_FILE} --train_cfg config/ithemal_train.json --model_cfg config/ithemal_model.json --experiment-name {EXPERIMENT_NAME} --experiment-time {EXPERIMENT_NAME}`
* `config/ithemal_train.json`: configuration of trainer
* `config/ithemal_model`: configuration of transformer model
* `{DATA_FILE}`: data file for training. Its 80% will be used to train.  
* `{EXPERIMENT_NAME}` and `{EXPERIMENT_TIME}`: training result are saved into saved/Experiment_name/Experiment_time`.  

## Result
Training result saved at `saved/Experiment_name/Experiment_time`
* `loss_report`: (epoch, time, average loss of epoch, learning rate)  .
* `validation_result`:  (predicted, actual), (average loss), (correct-75%, total validataion data).
* `trained.mdl`: trained model.
* `predictor.dump`: model and data form of training.
* `checkpoints`: trained model per epoch.

# Validataion
To validate model, use `validation.sh`
It print validation result of 75%, 90%, 95% each

## validation.sh
`python validation.py --model saved/{EXPERIMENT_NAME}/{EXPERIMENT_TIME}/predictor.dump --model-data saved/{EXPERIMENT_NAME}/{EXPERIMENT_TIME}/trained.mdl --data {DATA_FILE}`
* `predictor.dump` and `trained.mdl`: output from training.
* `{DATA_FILE}`: same data file with training. Its 20% will be used to validation.
