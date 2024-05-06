

def get_eval_data_blob_name(
    bucket_uri = "gs://jskim/",
    blob_name = "dailyemail_v4_7_random_test/test_data_v2_jan_2023",
    max_blob_num = None):
    """
    Read parquet files from the blobs and return a pdanda dataframe.
    """
    eval_blob_list = bq.list_blobs("jskim", blob_name)
    eval_blob_list = [bucket_uri + b for b in eval_blob_list]
    print(f"Total evaluation blobs number: {len(eval_blob_list)}")

    if max_blob_num:
        if max_blob_num > len(eval_blob_list):
          max_blob_num = len(eval_blob_list)
        eval_blob_list = eval_blob_list[:max_blob_num]

    eval_dataset = pd.DataFrame()

    for b in eval_blob_list:
      tmp = pd.read_parquet(b)
      eval_dataset = pd.concat([eval_dataset, tmp], axis=0)

    return eval_dataset



