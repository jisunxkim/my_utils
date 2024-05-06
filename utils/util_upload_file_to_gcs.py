#!/usr/bin/env python3

from google.cloud import storage
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', help='local file path')
    parser.add_argument('blob_name', help='blob name')
    parser.add_argument('--bucket_name', default='jskim', help='bucket name')
    parser.add_argument('--project_name', default='zulilymodeltraining', help='GCP project name')
    
    args = parser.parse_args()
    
    project_name = args.project_name
    bucket_name = args.bucket_name
    blob_name = args.blob_name
    local_file_path = args.file_path

    storage_client = storage.Client(project=project_name)

    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_file_path, timeout=5000)
        print(f"Succesfully uploaded {local_file_path} to {blob}")
    except Exception as e:
        print(f"Failed to upload {local_file_path}")
        print(e)

