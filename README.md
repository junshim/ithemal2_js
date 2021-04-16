# Overview
Each file contains the modified model based on [**the original ithemal source code**](https://github.com/ithemal/Ithemal)  for each purpose below.
More detailed explanations and training methods for each model exist in each folder.  
   
### ithemal_LSTM
Base line model which use LSTM.
Since the original ithemal source code is optimized for cpu, I is modified to run on gpu.
  
### ithemal_transformer_two_layer
It is a model using transformer model, not LSTM.  
The two layer means that the instruction layer and the basic block layer are divided.  
>   Two layer does not mean the number of encoder layers of the transformer.
    
### ithemal_transformer_one_layer
It is a model that combines the two layers mentioned above into one layer.
