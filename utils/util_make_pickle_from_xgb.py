#!/usr/bin/env python3

import xgboost as xgb
import pickle
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_file", help="file name to make a pickle")
    parser.add_argument("dest_file",
                        default = "converted_pickle_file", help= "destination file name")

    args = parser.parse_args()

    # Read the content from the text file
    file_path = args.source_file # Replace with your file path
    dest_file_name = args.dest_file

    try:
        model = xgb.Booster()
        model.load_model(file_path)
        print("loaded xgboost model")
    except Exception as e:
        print("Failed to load the xgboost model")
        print(e)
    # Convert the content to a pickle file
    try:
        pickle_path = f"{dest_file_name}.pkl"  # Replace with your desired pickle file path
        with open(pickle_path, "wb") as pickle_file:
            pickle.dump(model, pickle_file)
        print(f"Succesfully wrote to pickle file: {pickle_path}")

    except Exception as e:
        print(f"Failed to write to pickle file: {pickle_path}")
        print(e)
