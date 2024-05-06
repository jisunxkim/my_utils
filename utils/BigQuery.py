from google.cloud import bigquery
from google.cloud import storage
from utils import gcloud_setup
import os
import subprocess
import pandas as pd



class BigQuery:
    def __init__(self, project_id=None):
        self.project_id = project_id
        self.status = None
        self.reset_project(project_id=project_id)
        self.bigquery_client = self.get_bigquery_client()
        self.storage_client = self.get_storage_clientt()
        
    def reset_project(self, project_id=None):
        """
        """
        if gcloud_setup.setup_project(project_id):
            self.project_id = os.environ['GOOGLE_CLOUD_PROJECT']
            self.status = True
        else:
            self.status = False
            print("Failed to set up bigquery project environment!!")
    
    def get_status(self):
        print(self.status)
        return self.status
    
    def get_gcloud_config(self):
        subprocess.run(['gcloud', 'config', 'list'])

    def get_bigquery_client(self):
        if not self.status:
            return None
        return bigquery.Client(project=self.project_id)
    
    def get_storage_clientt(self):
        if not self.status:
            return None
        return storage.Client(project=self.project_id)
    
    def get_jobs(self, state = "DONE", n_jobs = 2, mins = None):
        """
        state: "DONE", "PENDING", "RUNNING"
        Return a list of google.cloud.bigquery.job.query.QueryJob object
        There are many useful properties and actions can do with the job object
        Example)
        jobs = get_jobs()
        job = jobs[0] # first job in the list
        job.query # query of the job
        job.created # daetime of the job craeted
        job.total_bytes_billed
        job.to_dataframe() # collect df from the job
        job.state # job status
        """
        if mins:
            m_mins_ago = datetime.datetime.utcnow() - datetime.timedelta(minutes=mins)
        else: 
            m_mins_ago = None

        job_list = []
        called_jobs = client.list_jobs(
                max_results=n_jobs, 
                min_creation_time=m_mins_ago, 
                state_filter=state)  # API request(s)
            
        for job in called_jobs:
            print("{}".format(job.job_id))
            job_list.append(job)

        return job_list

    
    def get_query(self, path=None, query=None, **kwargs):
        """
        Load a query from a file or strings.
        """
        if path:
            try:
                with open('{}'.format(path)) as f:
                    q = f.read()
            except Exception as e:
                print(f"!!!Cannot open the file {path}")
                return None

        elif query:
            q = query
        else: 
            print("!!!Provide a query file path or query")
            return None

        if len(kwargs) > 0:
            q = q.format(**kwargs)

        return q
    
    def query_dry_run(self, query, max_GB=0.5, print_query=True):
        
        print("*"*40)
        print("running dry-run....")
        if print_query:
            print("query to run:")
            print(query)
        try:
            max_bytes = int(max_GB * 1e9)
            

            # job configuration
            job_config = bigquery.QueryJobConfig()
            job_config.maximum_bytes_billed = max_bytes
            job_config.allow_large_results = True

            # make API request
            job = self.bigquery_client.query(
                query, 
                job_config=job_config)

            # Wait for the job to complete.
            job.result() 

            # get data size
            data_size_mb = job.total_bytes_processed / 1e6

            print("*"*10)
            print("*"*3, "Dry Run Passed!!!!", "*"*3)
            print(f"data processed: {data_size_mb:0.4f} MB")
            print(f"maximum size set: {max_GB:0.4f} GB")
            print("*"*20)
            
            return True

        except Exception as e:
            print("*"*20)
            print("*"*3, "Dry Run FAILED!!!!", "*"*3)
            print(f"maximum size set: {max_GB:0.1f} GB")
            print(str(e))
            print("*"*40)
            
            return False
    
    def query_to_table(self
                       , query
                       , dataset_id
                       , table_id
                       , project_id=None
                       , write_option ="WRITE_EMPTY"
                       , after_dry_run=True
                       , max_GB=0.5
                       , print_query=True
                      ):
        """
        Run a query and save the results as a table
        write_option: "WRITE_TRUNCATE", "WRITE_APPEND", "WRITE_EMPTY"
        WRITE_EMPTY: If the table already exists and contains data, 
                    a 'duplicate' error is returned in the job result.
        """
        if not project_id:
            project_id=self.project_id
        
        if after_dry_run:
            if self.query_dry_run(query,max_GB=max_GB):
                run_query = True
            else:
                run_query = False
                print("!!!Stopping query to table job due to fail in Dry Run!!!")
                return None
        else:
            run_query = True
        
        if run_query:
            try: 
                max_bytes = int(max_GB * 1e9)
                
                # Create reference to the destination table
                table_ref = self.bigquery_client\
                    .dataset(dataset_id)\
                    .table(table_id)
                print(f"destination table_ref:{table_ref}") 

                job_config = bigquery.QueryJobConfig()
                job_config.destination = table_ref
                job_config.destination.projectID = project_id
                job_config.maximum_bytes_billed = max_bytes
                job_config.allow_large_results = True
                job_config.write_disposition = write_option

                query_job = self.bigquery_client.query(query, job_config=job_config)
                query_job.result()  # Waits for the query to finish
                print("*"*40)
                print('Query results were written to the table {}'.format(table_ref.path))
                print("*"*40)

                return table_ref

            except Exception as e:
                print("*"*40)
                print('!!!FAILED to query and write to the table {}'.format(table_ref.path))
                print(e)
                print("*"*40)
                return None
            
    
    def query_to_df(self, query,
                    max_GB=0.5,
                    after_dry_run=True,
                    print_query=False,
                   ):
        
        if after_dry_run:
            if self.query_dry_run(query,max_GB=max_GB, print_query=print_query):
                run_query = True
            else:
                run_query = False
                print("!!!Stopping query to table job due to fail in Dry Run!!!")
                return None
        else:
            run_query = True
        
        if run_query:
            
            try:
                max_bytes = int(max_GB * 1e9)
                job_config = bigquery.QueryJobConfig()
                job_config.maximum_bytes_billed = max_bytes
                job_config.allow_large_results = True
                
                job = self.bigquery_client.query(
                    query, 
                    job_config=job_config
                )
                
                job.result()  # Waits for the query to finish
                data_size_mb = job.total_bytes_processed / 1e6
                print("*"*40)
                print('Query succesfully done. Returning as dataframe...')
                print(f"data processed: {data_size_mb:0.1f} MB")
                print("*"*40)
                      
                return job.to_dataframe()      

            except Exception as e:
                print("*"*40)
                print('!!!FAILED to query and return to a dataframe.')
                print(e)
                print("*"*40)
                
                return None
    
    def get_columns_types(self, dataset, table_name, project=None):
        if not project:
            project = "zulilymodeltraining"

        query = f"""
        SELECT column_name, data_type
        FROM {project}.{dataset}.INFORMATION_SCHEMA.COLUMNS
        WHERE table_name = '{table_name}'
        """
        df = self.query_to_df(query,after_dry_run=False)
        
        return df
            
            
    def table_to_gcs(self,
                     s_project, s_dataset, s_table_id,
                     bucket_name, blob_name,
                     blob_name_suffix ='',
                     destination_format = 'Parquet',
                    ):
        """
        Export BigQuery table to a GCS bucket
        destination_format: "CSV, "JSON", "Parquet"
        
        You can export up to 1 GB of table data to a single file. 
        If you are exporting more than 1 GB of data, use a wildcard 
        to export the data into multiple files. When you export data 
        to multiple files, the size of the files will vary. 
        To control the exported file size, 
        you can partition your data and export each partition.
        Example)
        bq.table_to_gcs(
            s_project='zulilymodeltraining'
            , s_dataset='dailyemail_v4_7_sample'
            , s_table_id='v4_7_sample_oct1_15_2022__very_small_target'
            , bucket_name='jskim'
            , blob_name='tesing_bq_codes'
            , blob_name_suffix="/*.parquet"
            , destination_format='Parquet')
        """

        destination_uri = (
            'gs://{}/{}{}'
            .format(bucket_name, blob_name, blob_name_suffix)
        )
        
        print("*"*40)
        print("Exporting {}:{}.{} to {} ........"
              .format(s_project, s_dataset, s_table_id, destination_uri)
             )
        
        try:
            dataset_ref = bigquery.DatasetReference(s_project, s_dataset)
            table_ref = dataset_ref.table(s_table_id)
            
            job_config = bigquery.ExtractJobConfig()
            job_config.destination_format = destination_format

            job = self.bigquery_client.extract_table(
                source=table_ref,
                destination_uris = destination_uri,
                job_config=job_config,
            )  
            
            job.result()  # Waits for job to complete.

            # job_config = bigquery.ExtractJobConfig()
            # job_config.destination_format = destination_format
            # job = self.bigquery_client.extract_table(table_ref, blob_uri, job_config=job_config)
            # job.result()  # Waits for job to complete.

            print("*"*40)
            print("Exported {}:{}.{} to {}"
                  .format(s_project, s_dataset, s_table_id, destination_uri)
                 )
            print("Returning destination uri")
            print("*"*40)
            
            return destination_uri
        
        except Exception as e:
            
            print("*"*40)
            print('!!!FAILED to extract the table to gcs.')
            print(e)
            print("*"*40)
            
    def list_blobs(self, bucket_name, blob_name):
        """
        Large query results are returned in multiple files.
            This function returns list of the splits
        Example:
        list first level of blub names including 'test' at full blub names
        set([b.split("/")[0] for b in bq.list_blobs("jskim", 'test')])
        
        """
        # List all the blobs in the bucket
        
        blobs = self.storage_client.list_blobs(bucket_name)
        # Return a filtered list of blobs
        return [b.name for b in blobs if b.name.find(blob_name) != -1]
            
    def download_blobs(self, 
                       bucket_name, 
                       blob_name,
                       chunk_size = 256 * 156,
                       verbose=True
                      ):
        """
        chunk_size: int, (Optional) The size of a chunk of data 
            whenever iterating (in bytes).
            This must be a multiple of 256 KB per the API specification. 
            If not specified, the chunk_size of the blob itself is used. 
            If that is not specified, a default value of 40 MB is used.
        """
        # Download the blob(s) from the specified bucket
        bucket = self.storage_client.get_bucket(bucket_name)
        
        for blob in self.list_blobs(bucket_name, blob_name):
            bucket.blob(f'{blob}').download_to_filename(blob.split('/')[-1])
            if verbose == True:
                print('Blob {} downloaded'.format(blob))
                
    
    def delete_blobs(self, 
                     bucket_name, 
                     blob_name,
                     show_error=False, 
                    ):
        
        bucket = self.storage_client.get_bucket(bucket_name)
        for blob in list_blobs(bucket_name, blob_name):
            
            try:
                bucket.blob(blob).delete()
                print(f"Succesfully deleted {blob}")
            except Exception as e:
                print(f"Failed to delete {blob}")
                if show_error:
                    print(e)
        print(f'Completed deleting {blob} from bucket {bucket_name}') 
        

    def upload_file_to_blob(self,
                            local_file_path, 
                            bucket_name, 
                            blob_name,
                            timeout_seconds = 5000
                           ):
        
    
        try:
            bucket = self.storage_client.get_bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_file_path, timeout=timeout_seconds)
            print(f"Succesfully uploaded {local_file_path} to {blob}")
        except Exception as e:
            print(f"Failed to upload {local_file_path}")
            print(e)
            
            
    def download_file_from_blob(self, bucket_name, source_blob_name, destination_file_name, verbose = 1):
        """
        # Usage example
        download_file("your-bucket-name", "path/to/your/file.txt", 
                        "/path/to/destination/local/file.txt")
        """
        bucket = self.storage_client.get_bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        if verbose > 0:
            print(f"File {source_blob_name} downloaded to {destination_file_name}")

    
            
    def get_tables_updated(self, project_tableset):
        
        query = f"""
        SELECT table_id, TIMESTAMP_MILLIS(last_modified_time)
        FROM `{project_tableset}.__TABLES__`
        order by last_modified_time desc
        """
        return pd.read_gbq(query)
    
    def df_to_table(self, dataframe, destination, write_method='WRITE_TRUNCATE'):
        """
        write dataframe to a bigquery table.
        """
        
        job_config = bigquery.LoadJobConfig(
            # write_option: "WRITE_TRUNCATE", "WRITE_APPEND", "WRITE_EMPTY"
            write_disposition = write_method
        )

        job = self.bigquery_client.load_table_from_dataframe(
            dataframe=dataframe, 
            destination=destination, 
            job_config=job_config
        )

        job.result()

        
        
        
    

            
