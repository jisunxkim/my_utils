#!/bin/bash

##### Get the current date and time in the format YYYYMMDD_HHMMSS
current_datetime=$(date +"%Y%m%d_%H%M%S")

## regular xgb
# log_file="./log/model_training_xgb_fv2_2_${current_datetime}.log"
# log_file="./log/train_experiment_xgbbooster_reload_${current_datetime}.log"
log_file="./log/train_test6_${current_datetime}.log"

jupyter nbconvert xgb_train_multiprocessing.ipynb --to python 
python3 -u xgb_train_multiprocessing.py ./config/xgb_train_config_test6.yaml
# nohup python3 -u xgb_train_multiprocessing.py ./config/xgb_train_config_test6.yaml > "$log_file" 2>&1 &

#python3 -u xgb_train_multiprocessing.py ./config/xgb_train_config01.yaml
#nohup python3 -u xgb_train_multiprocessing.py ./config/xgb_train_config01.yaml > "$log_file" 2>&1 &

### ray xgb
#log_file="./log/temp_model_training_ray_xgb_fv2_2_${current_datetime}.log"
# jupyter nbconvert ray_xgb_train_multiprocessing.ipynb --to python 
# python3 -u ray_xgb_train_multiprocessing.py ./config/ray_xgb_train_config_test.yaml
#nohup python3 -u ray_xgb_train_multiprocessing.py ./config/ray_xgb_train_config_test.yaml > "$log_file" 2>&1 &  
