#!/usr/bin/env python3

import xgboost as xgb
from xgboost import XGBClassifier
import pickle
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="file name to make a pickle")

    args = parser.parse_args()

    # Read the content from the text file
    file_path = args.file_name # Replace with your file path

    # model = xgb.Booster()
    model = xgb.XGBClassifier()
   
    model.load_model(file_path)
    # Convert the content to a pickle file
    pickle_path = f"{file_path.split('.')[0]}.pkl"  # Replace with your desired pickle file path
    with open(pickle_path, "wb") as pickle_file:
        pickle.dump(model, pickle_file)

