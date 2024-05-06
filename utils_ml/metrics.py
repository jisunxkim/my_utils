import numpy as np
import pandas as pd
import logging

import multiprocessing

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import ndcg_score

from matplotlib import pyplot as plt

from google.cloud import bigquery

bqclient = bigquery.Client()

def get_logger(name=__name__, level="INFO", clear=False):
    """
    # level: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
    example:
    logger = get_logger(level="INFO", clear=True)
    ### remove handlers
    logger.handlers.clear()
    """

    # Create logger
    logger = logging.getLogger(name)
    
    # Clear handler
    if clear:
        logger.handlers.clear()
        
    if not logger.hasHandlers():
        # Create log handler
        log_handler = logging.StreamHandler()
        log_handler.setLevel(level)

        # Set handler format
        log_format = logging.Formatter("%(asctime)s %(name)s[%(levelname)s]: %(message)s",
                                       datefmt='%m/%d/%Y %I:%M:%S %p')
        log_handler.setFormatter(log_format)

        # Add handler to logger
        logger.addHandler(log_handler)
        
    logger.setLevel(level)

    return logger


def print_metrcs_inline(eval_Y, y_pred, loss, best_iter):
    """
    Performance metrics of each batch training.
    """
    # evaluate predictions
    accuracy = accuracy_score(eval_Y, y_pred)
    balanced_accuracy_value = balanced_accuracy_score(eval_Y, y_pred)
    f1_value = f1_score(eval_Y, y_pred)
    recall_val = recall_score(eval_Y, y_pred)

    min_loss = min(loss)
    last_loss = loss[-1]
    epochs = len(loss)

    print(
      "| Accu: %.2f%%" % (accuracy * 100.0), \
      "| Bal Accu: %.2f%%" %(balanced_accuracy_value * 100.0), \
      "| Recall Pos: %.3f" %(recall_val), \
      "| F1 Pos: %.3f" %(f1_value), \
      "| Epochs: %.0i" %(epochs), \
      "| Best Iter Num: %.0i" %(best_iter), \
      "| Min loss: %.3f" %(min_loss), \
      "| Last loss: %.3f" %(last_loss) \
    )

def plot_eval_metric(
    title,
    eval_results, 
    best_iteration, 
    metric_name='logloss',
    safe_file=False):
    """
    parameters
        eval_results is an object from trained_model.evals_results() where a model should be trained using eval_set_list 
    """
    
    # plot log loss
    epochs = len(eval_results['validation_0'][metric_name])
    x_axis = range(0, epochs)
    fig, ax = plt.subplots()
    ax.plot(x_axis, eval_results['validation_0'][metric_name], label='Train')
    ax.plot(x_axis, eval_results['validation_1'][metric_name], label='Test')
    ax.legend()
    # plt.axvline(x = best_iteration, color = 'b', ls='--')
    plt.xlabel("boosting rounds(num of estimators or subtrees")
    plt.ylabel(metric_name)
    plt.title(f'XGBoost {metric_name}')
    plt.show()

    
def get_sort_scores(name, treatment):
    """
    treatment: dict, keys=["table_name", "score_column", "truth_column"]
    """
    process = multiprocessing.current_process()
    logger = get_logger(name=process.name)
    
    logger.info(f"Loading prediction table of {name}")
    
    df = pd.DataFrame()

    try:
        pred_table = treatment["table_name"]
        score_column = treatment["score_column"]
        truth_column = treatment["truth_column"]

        query = f"""
        SELECT DISTINCT 
            pred.as_of_date, pred.customer_id, pred.event_id,  
            CAST(pred.{score_column} AS FLOAT64) as score,
            CAST(pred.{truth_column} AS INT64) as truth_label
        FROM `{pred_table}` pred
        """
        df = bqclient.query(query).to_dataframe()

        df['Treatment'] = name

        df['as_of_date'] = pd.to_datetime(df['as_of_date'])
        df['score'] = df['score'].astype(float)
        df['truth_label'] = df['truth_label'].astype(int)
        
    except Exception as e:
        logger.critial(e)
    
    return df

def get_sort_scores_multi_processing(treatments, n_cpu = 1, label_threshold = 0.5):
    """
    label_threshold: probability threshold for positive or negative
    treatments: dictionary of treatment name, {table_name, score_column,truth_column }
    ex)
    treatments = {
            "xgb_medium_plus_08262023_010402":
            {
                "table_name":  'zulilymodeltraining.dailyemail_v4_7_prediction.medium_plus_xgb_train_model_08262023_010402_best_tuned_test_data__v2_2_jan_2023',
                "score_column": 'pos_score',
                "truth_column": 'label_578'
            }, 
        }
    
    """
    if not isinstance(treatments, dict):
        print("Error: treatments should be a dictionary!")
        return 
    
    n_cpus = len(treatments)
    n_cpus

    try:
        p = multiprocessing.Pool(processes=n_cpus)
        scores_list = p.starmap(func=get_sort_scores, iterable=treatments.items())

        p.close()
        p.join()

        scores_df = pd.concat(scores_list, axis=0)
        scores_df['pred_label'] = np.where(scores_df.score > label_threshold, 1, 0)
    except Exception as e:
        print(f"Failed to load prediction table by multiprocessing")
        print(e)
    
    return scores_df


def get_hr_ndcg(df, 
                COLUMNS_TO_RANKS = ['as_of_date', 'customer_id'], 
                K = 10, log_level='INFO'):
    """
    parameters:
        COLUMNS_TO_RANKS = ['as_of_date', 'customer_id']: group coloumns to calculate metrics for
        K = 10: top rank
    """
    
    model_metrics = pd.DataFrame()
    
    treatments = df.Treatment.drop_duplicates().values

    for trt in treatments:
    # for trt in ['Model1']:
    
        print("*"*30)
        print(f"Calculating for: {trt}")
        tmp = pd.DataFrame()
        
        if trt == 'perfect_model':
            ignore_ties = True
        else:
            ignore_ties = False
            
        
        tmp_hit, tmp_ndct = get_ltr_metrics(
            model_results_df = df[df.Treatment == trt], 
            K=K,
            target_column='truth_label',
            score_column='pred_label', # default: 'score'
            columns_to_rank=COLUMNS_TO_RANKS,
            log_level=log_level,
            # ignore_ties = ignore_ties
            ignore_ties = True
        )
        tmp.loc[:, 'Ranks'] = range(1, K+1)
        tmp.loc[:, 'Treatment'] = trt
        tmp.loc[:, 'hit_rate'] = tmp_hit
        tmp.loc[:, 'ndcg'] = tmp_ndct

        model_metrics = pd.concat([model_metrics, tmp], axis=0)
    
    return model_metrics


def get_ltr_metrics(
    model_results_df, K, 
    target_column, 
    score_column, 
    columns_to_rank = 'customer_id',
    log_level = 'INFO',
    ignore_ties = True):
    
    """
    Learning To Raking (LTR)
    Returns two LTR metrics: hit rate and NDCG metric
    Basically implements the evaluation in https://arxiv.org/abs/1708.05031
    The NDCG evaluation details can be found in http://staff.ustc.edu.cn/~hexn/papers/cikm15-trirank-cr.pdf
    
    :param model_results_df: Dataframe containing a customer_id column, target column and a model score column . Assumes that for each
    customer, at laest K items are present to be ranked.
    :param K: Evaluates the metrics for lists of length 1 to K
    :param target_column: Name of the column containing the target values
    :param score_column: Name of the column containing the model scores
    
    :returns:
        hit_rates: array of length K containing the hit rates for lists of length 1 to K
        ndcg_scores: array of length K containing the NDCG scores for lists of length 1 to K
        
    sklearn.metrics.ndcg_score(y_true, y_score, *, k=None, sample_weight=None, ignore_ties=False)
        ignore_ties: bool, default=False, Assume that there are no ties in y_score (which is likely to be the case if y_score is continuous) for efficiency gains.
    """
    logger = logging.getLogger()
    logger.setLevel(level=log_level)
    
    columns_to_rank = (
        columns_to_rank 
            if isinstance(columns_to_rank, list) 
            else [columns_to_rank]
    )

    # Group results by customer_i or columns_to_rank, and rank the results by score
    # ranks start from 1 to K
    model_results_df.loc[:, 'ranks'] = (
        model_results_df.groupby(by=columns_to_rank)[score_column]
        .transform(lambda x: x.rank(ascending=False, method='first', na_option='bottom'))
    )
    
    logger.debug("model_results_df \n" + str(model_results_df.head(2)))

    data_less_than_K = (
        model_results_df
        .groupby(columns_to_rank)
        .filter(lambda x: len(x) < K)
        .index.values
    )
    if len(model_results_df) == 0:
        print("ERROR empty model dataframe: Check model data table.")
        return 
    
    logger.info(
        f"""There are {len(data_less_than_K)} rows out of total {len(model_results_df)}, \
        {round(len(data_less_than_K)/len(model_results_df) * 100, 1)} % , \
        that less than number of ranks, {K}, by gruping {columns_to_rank}.""")

    logger.info("Removing those data....")
    model_results_df = model_results_df.drop(data_less_than_K).copy()

    # Select the top K results for each customer
    model_results_filtered_df = model_results_df[model_results_df['ranks']<=K].copy()

    # Find number of unique data of columns_to_rank in dataset
    unique_data = model_results_filtered_df[columns_to_rank].drop_duplicates().copy()
    num_unique_data = len(unique_data)

    logger.info(f"Total unique {columns_to_rank} selected for NDCG: {num_unique_data}")
    
    logger.debug("model_results_filtered_df\n" + str(model_results_filtered_df.head(2)))
    
    sample_customer = model_results_filtered_df.customer_id.values[0]
    logger.debug(
        f"model_results_filtered_df customer: {sample_customer}\n" + \
        str(model_results_filtered_df[model_results_filtered_df.customer_id == sample_customer]\
            .sort_values(['as_of_date','ranks'])))
    

    # Get hit rate (how many positive engagement are in top X)
    model_hit = list()
    for kk in np.arange(1, K+1):
        model_rank_sub = model_results_filtered_df[model_results_filtered_df['ranks'] <= kk].copy()
        hits = np.sum(model_rank_sub[target_column]>0)
        model_hit.append(hits/num_unique_data)    

    # Calculate NDCG Metric
    # Pivot to create a matric num_customers X K ranks of the scores
    y_score = model_results_filtered_df.pivot(index=columns_to_rank, columns = 'ranks', values=score_column)
    y_true = model_results_filtered_df.pivot(index=columns_to_rank, columns = 'ranks', values=target_column)

    # Note that in the pivot above, the rank just makes sure that there is a one to one correspondence between items in the y_score
    # and y_true matrices for each customer or columns_to_rank. The NDCG uses the y_true matrix to get relevance and true-rank information. y_score 
    # information is used to just get the predicted rank information. 

    # Get NDCG Metric
    model_ndcg = list()
    
    for kk in np.arange(1,K+1):
        ndcg_val = ndcg_score(y_true, y_score, ignore_ties = ignore_ties, k=kk)
        model_ndcg.append(ndcg_val)     

    return model_hit, model_ndcg


