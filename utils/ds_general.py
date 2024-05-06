import datetime
import pytz
import time
import yaml
import sys
import os
import logging
import pandas as pd
from itertools import islice
from tqdm import tqdm
import multiprocessing


def read_parquets_multi_processing(blobs_list, n_cpu = 1, n_files_per_cpu=1):
    """
    Multi processing of loading model data including data cleaning.
    Return: outputs of make_train_data()
        
    """

    
    try: 
        print(f"Data loading total {len(blobs_list)} blobs starting {blobs_list[0]}...")
        
        splited_loading_list = split_list_n_size(
                blobs_list, 
                n_files_per_cpu)
        
        p = multiprocessing.Pool(n_cpu)
        pool_results = p.map(
                func=read_parquet_from_blob, 
                iterable=splited_loading_list)
        p.close()
        p.join()
        
        data_df = pd.concat(pool_results, axis=0)
        
        df_info(data_df, label="dataframe from the blobs")
        
        return data_df
        
    except Exception as e:
        msng = f"Failed in geting model data by multi processing starting {blobs_list[0]}."
        print(e)
    

def read_parquet_from_blob(blob_path):
    """
    Load model data from a blob.
    Return a data frame
    """
    
    process = multiprocessing.current_process()
    logger = get_logger(name=process.name)
    
    temp_df = pd.DataFrame()
    result_df = pd.DataFrame()
    try:
        if type(blob_path) == str:
            blob_path = [blob_path]
        for i, blob in enumerate(blob_path):
            logger.info(f"{i+1}th blob loading......")
            temp_df = pd.read_parquet(blob)
            temp_df = temp_df.reset_index(drop=True)
            result_df = result_df.reset_index(drop=True)
            result_df = pd.concat([result_df, temp_df], axis=0)
        
        return result_df
            

    except Exception as e:
        msg = "Failed to load the blob."
        print(msg)
        print(e)

def write_list_to_file(file_path, data_list):
    with open(file_path, 'w') as file:
        for item in data_list:
            file.write(str(item) + '\n')

def read_list_from_file(file_path):
    with open(file_path, 'r') as file:
        data_list = [line.strip() for line in file]
    return data_list


def make_folder(folder_name):
    # Create a folder using a relative path (starting with a dot)
    if not os.path.exists(f"./{folder_name}"):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully.")
    else:
        print(f"Folder '{folder_name}' already exists.")
        
def sort_dict_by_val(my_dict, descending=True):
    return {
            k: v for k, v \
            in sorted(
                my_dict.items(), 
                key=lambda item: item[1], 
                reverse=descending
            )}
    
def get_n_items_dict(my_dict, num_items):
    """
    return first n (num_items) items from a dictionary.
    """
    return dict(islice(my_dict.items(), num_items))


def terminate_prog(msg, error = None):
    print("*"*30)
    print("*"*30)
    print("*"*30)
    print(f"CRITICAL...Terminating the program: {msg}")
    print(error)
    sys.exit(1)

class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.line_buffer = ""

    def write(self, message):
        # Log the message as if it were an error
        self.line_buffer += message
        if '\n' in self.line_buffer:
            lines = self.line_buffer.split('\n')
            for line in lines[:-1]:
                self.logger.log(self.level, line)
            self.line_buffer = lines[-1]

    def flush(self):
        pass

def get_logger(name=__name__, level="INFO", log_file = None, filemode="a", clear=False, all_print_log=True):
    """
    filemode="a",  # Use 'a' to append to the log file, 'w' to overwrite
    # level: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
    example:
    logger = get_logger(level="INFO", clear=True)
    ### remove handlers
    logger.handlers.clear()
    """
    
    if level.lower() == 'info':
        level = logging.INFO
    elif level.lower() == 'debug':
        level = logging.DEBUG
    elif level.lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO
    
    # Create a logger
    logger = logging.getLogger(name)
    
    # Clear handler
    if clear:
        logger.handlers.clear()

    # Set handler format
    log_format = logging.Formatter("%(asctime)s %(name)s[%(levelname)s]: %(message)s",
                                   datefmt='%m/%d/%Y %I:%M:%S %p')
    
    if log_file:
        # Create a file handler to log messages to the log file
        file_handler = logging.FileHandler(filename=log_file, mode=filemode)
        file_handler.setLevel(level)
        file_handler.setFormatter(log_format)
        
        # Add the file handler to the logger
        logger.addHandler(file_handler)

    # Create a stream handler to log messages to the terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(log_format)

    # Add the sream handler to the logger
    logger.addHandler(stream_handler)
    
    if all_print_log:
        # pass
        sys.stdout = LoggerWriter(logger, level)

    logger.setLevel(level)
    
    return logger

def split_list_n_size(lst, n_size):
    if not isinstance(lst, list) or not lst:
        raise TypeError("a list should be provided to split") 
    
    if len(lst) < 2:
        return lst
    
    splited_list = [
        lst[i:(i+n_size)]
        for i in range(0, len(lst), n_size)
    ]
    
    return splited_list

    
def time_start_end(started=None, print_time=True, msg=""):
    
    if not started:
        start_time = datetime.datetime.now(pytz.timezone('US/Pacific'))
        
        if print_time:
            print(msg + " Starting at ", start_time.strftime('%m%d_%H:%M:%S'), "*"*50)
        return start_time
    else: 
        end_time = datetime.datetime.now(pytz.timezone('US/Pacific'))
        time_took = round((end_time - started).total_seconds() / (60*60), 2)
        time_took_min = round((end_time - started).total_seconds() / (60), 2)
        time_took_sec = round((end_time - started).total_seconds(), 0)
        if print_time:
            print("*"* 80)
            print(f"{msg} Ending at", end_time.strftime('%m%d_%H:%M:%S'), \
                  f"took {time_took} hour", f"or {time_took_min} min", f"or {time_took_sec} sec")
        return time_took
    
def time_id(
    t_now=None, 
    time_zone = 'US/Pacific', 
    time_format = '%m%d%Y_%H%M%S'):
    """
    parameters:
        t_now: datetime.datetime.now()
    """
    if not t_now:
        t_now = datetime.datetime.now(pytz.timezone(time_zone)) 
    return t_now.strftime(time_format)

def name_time_id(name = ""):
    return name+'_'+time_id()

def load_config(fname, path=None):
    if path:
        f_path = os.path.join(path, fname)
    else:
        f_path = fname
        
    with open(f_path) as file:
        config = yaml.safe_load(file)

    return config

### pandas related

def find_all_zero_cols(df):
    df = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32'])
    return df.columns[(df==0).all()].values

def get_cols_quantile_value(df, quantile, threshold_value, less_than=True, numeric_only=True):
    """
    return columns whose percentile value is less than or equal to the threshold_value.
    """
    df = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32'])
    temp = df.quantile(q=quantile, numeric_only=True)
    if less_than:
        return temp[temp <= threshold_value].index.values
    
    else:
        return temp[temp >= threshold_value].index.values

def save_columns_file(column_index, file_name):
    """
    Save list of columns from pandas comlumn index 
    to a file for a review and selections for 
    training.
    """
    with open(f'{file_name}.txt', 'w') as file:
        for column_name in column_index:
            file.write(column_name + '\n')

def df_info(df, label = "dataframe Info", return_dic=False):
    val = [
        df.shape, 
        df.dropna().shape, 
        round(df.memory_usage(deep=True).sum() / 1e6, 2),
        df.dtypes.value_counts()
    ]
    labels=[
        "shape",
        "dropna_shape",
        "size_MB",
        "dtypes"
    ]
    
    temp = dict(zip(labels, val))
    if label:
        print("*"*3, "df:", label, "*"*3)
    for item in temp:
        if item == 'dtypes':
            print(item, ":")
            print(temp[item])
        else:
            print(item, ":", temp[item])
    if return_dic:
        return temp

# Bigquery storage related
def test_blob_loading (
    blob_list,
    batch_size = 2,
    total_size = 5,
    verbose = 1):
    """
    Test size and time duration of loading
    blob parquet files to pandas dataframe
    in batchs. 
    Parameters:
        blob_list: list of blobs, blob should be a full name like "gs://bucket/blob"
        total_size: number of total files to load
            set None to load all blobs.
        batch_size: number of blobs to load in each batch
    """

    s_time = time_start_end()

    if not total_size:
        total_size = len(blob_list)

    n_iter = 0
    mem_size, num_rows, num_blobs = 0, 0, 0

    for i in range(0, total_size, batch_size):
        temp_df = pd.DataFrame()

        if verbose > 0:
            print("** batch:",n_iter+1)
            n_iter += 1

        for j in range(i, i + batch_size if i + batch_size < total_size else total_size, 1):
            blob = blob_list[j]
            if verbose > 1:
                print(blob)

            temp_df = pd.concat([temp_df, pd.read_parquet(blob)], axis=0)
            num_blobs += 1
        mem_size += round(temp_df.memory_usage().sum() / 1e6, 0)
        num_rows += temp_df.shape[0]

    print("total num of batchs:", n_iter)
    print("total num of blobs:", num_blobs)
    print("total memory in MB:", mem_size)
    print("total rows:", num_rows)
    print("avg df memory size each batch in MB:", mem_size / n_iter)
    time_start_end(started=s_time)
    del temp_df