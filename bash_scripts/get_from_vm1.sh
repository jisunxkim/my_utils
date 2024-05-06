# !/bin/bash


## Train files
# gcloud compute scp jskim-vm1:~/dailyemail_model_training/xgb_model_train/config/xgb_train_config01.yaml --zone=us-central1-c ./config/
# gcloud compute scp jskim-vm1:~/dailyemail_model_training/xgb_model_train/xgb_train_multiprocessing.ipynb --zone=us-central1-c .
# gcloud compute scp jskim-vm1:~/dailyemail_model_training/xgb_model_train/run_train.sh --zone=us-central1-c .

gcloud compute scp jskim-vm1:~/dailyemail_model_training/xgb_model_train/config/xgb_ray_train_config01.yaml --zone=us-central1-c ./config/
gcloud compute scp jskim-vm1:~/dailyemail_model_training/xgb_model_train/xgb_ray_train_multiprocessing.ipynb --zone=us-central1-c .
# gcloud compute scp jskim-vm1:~/dailyemail_model_training/xgb_model_train/run_train.sh --zone=us-central1-c .

