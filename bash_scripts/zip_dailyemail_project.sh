# !/bin//bash

ex1=$(find ./dailyemail_model_training -type d -name ".*" -prune)
ex2=$(find ./dailyemail_model_training -type d -name "tmp")
ex3=$(find ./dailyemail_model_training -type d -name "temp")
ex4=$(find ./dailyemail_model_training -type d -name "data")
ex4=$(find ./dailyemail_model_training -type f -size "+3M")

zip -r dailyemail_model_training.zip ./dailyemail_model_training -x $ex1 -x $ex2 -x $ex3 -x $ex4