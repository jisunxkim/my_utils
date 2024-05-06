import subprocess
import os
import argparse

def setup_project(project_id):
    if not project_id:
        print("No project_id provided. Setting up for default.")
        project_id = "zulilymodeltraining"

    try: 
        result = subprocess.run(
            ['gcloud', 'config', 'set', 'project', project_id]
            , check=True # raise except when failed
            # , stderr=subprocess.STDOUT # combine stderr message into stdout
            , capture_output=True
            , text=True
        )

        os.environ['GOOGLE_CLOUD_PROJECT'] = project_id

        subprocess.run(['gcloud', 'config', 'list'])
        print("GOOGLE_CLOUD_PROJECT:", os.environ['GOOGLE_CLOUD_PROJECT'])

    except subprocess.CalledProcessError as e:
        print(f"Failed to set gcloud project: {project_id}")
        print(e)
        
        return False
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Set google cloud projet."
    )
    parser.add_argument("--project_id")
    args = parser.parse_args()
    
    # print(args.project_id)
    setup_project(project_id=args.project_id)
    
