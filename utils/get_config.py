def get_config(fname, path="../model_train"):
    """
    Load config YAML file. 
    fname: file name w/ or w/o ".yml"
    path: path to the file.
    """
    import os
    import yaml

    fname = fname.replace(".yml", "")

    config_file = f"{fname}.yml"
    config_file = os.path.join(path, config_file)
    
    if(os.path.exists(config_file)):
        try:
            with open(config_file, 'r') as f:
                conf = yaml.load(f, Loader=yaml.SafeLoader)
            
            print("Succesfully loaded: ", config_file)
            return conf

        except Exception:
            print("Error: check yaml syntax, key, and values in the config file!!")
            return False

    else:
        print(f"Error: file not exist, {config_file}")
    
    