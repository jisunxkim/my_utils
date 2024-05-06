import joblib
import gcsfs
    
    

def save_to_gcs_as_joblib(
    object_to_save,
    blob_name,
    file_name,
    bucket_name='jskim',
    remove_temp=True):
    """
    Save tunning, training, evalution model objects to gcs as joblib.
    """
    
    temp_file =  "temp_" + file_name
    joblib.dump(object_to_save, temp_file)

    # Copy the local model file to GCS and remove the local file
    bucket_name = 'jskim'
    blob_path = blob_name + "/" + file_name
    
    fs = gcsfs.GCSFileSystem()

    try:
        with fs.open(f'{bucket_name}/{blob_path}', 'wb') as blob_file:
            blob_file.write(open(temp_file, 'rb').read())
        
        if remove_temp:
            os.remove(temp_file)
        
        print(f"Completed moving {file_name} to gcs.")
              
    except Exception as e:
        print("Failed to save model to gcs!")
        print(e)