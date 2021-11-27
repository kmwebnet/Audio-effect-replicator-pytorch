# Audio-effect-replicator-pytorch
replicate audio effects by LSTM. based on [coz-a/Audio-Effect-Replicator](https://github.com/coz-a/Audio-Effect-Replicator).

# what is it  
Train audio effects by setting dry sound as X, effected sound as Y and predict audio effect for new given sound file.  

# Contents  
train.py  -- training main program  
config.yml -- define parameters and the list of sound file.    
predict.py -- predict by using trained model and out .nnp file.  
fx_replicator.py -- helper program for train.py and predict.py  

# Environment
sudo pip3 install -r requirements.txt

# Usage  

training  
python3 train.py   

You can use the tensorboard to monitor your learning status.  

inferencing using training model   
python3 predict.py  -i ./data/testdata1-2.wav -m checkpoint\date and time\best_result.pth     

you need to prepare WAV file as monoural 32bit signed int(uncompessed)format.  

# Exercise
Using the created training model,   
Inference operation can be performed not only on a PC with high processing power   
but also on Jetson Nano and Raspberry pi.  