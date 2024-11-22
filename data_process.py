from collections import defaultdict
import pandas as pd
import configparser
import sys
from sklearn.preprocessing import StandardScaler

def clean_data(df, config, dataset):
    # CLEAN data -- only keep columns as specified by the config file
    selected_columns = config[dataset].getlist("columns")

    # Bucketize text data
    text_columns = config[dataset].getlist("text_columns", [])
    for col in text_columns:
        # Cat codes is the 'category code'. Aka it creates integer buckets automatically.
        df[col] = df[col].astype('category').cat.codes

    # Remove the unnecessary columns. Save the variable of interest column, in case
    # it is not used for clustering.

    # Convert to float, otherwise JSON cannot serialize int64
    for col in df:
        if col in text_columns or col not in selected_columns: continue
        df[col] = df[col].astype(float)

    if config["DEFAULT"].getboolean("describe_selected"):
        print(df.describe())

    return df

def scale_data(df):
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df[df.columns]), columns=df.columns)
    return df

def subsample_data(df, N):
    return df.sample(n=N).reset_index(drop=True)

def take_by_key(dic, seq):
    return {k : v for k, v in dic.items() if k in seq}

def read_list(config_string, delimiter=','):
    config_list = config_string.replace("\n", "").split(delimiter)
    return [s.strip() for s in config_list]

if __name__ == '__main__':

    config_file = "config/example_config.ini"
    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(config_file)

    # Create your own entry in `example_config.ini` and change this str to run your own trial
    config_str = "hmda" if len(sys.argv) == 1 else sys.argv[1]

    print("Using config_str = {}".format(config_str))
    data_dir = config[config_str].get("data_dir")
    dataset = config[config_str].get("dataset")
    clustering_config_file = config[config_str].get("config_file")
    num_clusters = list(map(int, config[config_str].getlist("num_clusters")))
    max_points = config[config_str].getint("max_points")

    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(clustering_config_file)
    # Read data in from a given csv_file found in config
    csv_file = config[dataset]["csv_file"]
    df = pd.read_csv(csv_file, sep=config[dataset]["separator"])

    # Subsample data if needed
    if max_points and len(df) > max_points:
        df = subsample_data(df, max_points)


    # Clean the data (bucketize text data)
    df = clean_data(df, config, dataset)



    # Select only the desired columns
    selected_columns = config[dataset].getlist("columns")
    df = df[[col for col in selected_columns]]
    df.fillna(df.median(), inplace=True)

    # Scale data if desired
    scaling = config["DEFAULT"].getboolean("scaling")
    if scaling:
        df = scale_data(df)


    # save data
    df.to_csv('data/CleanData/'+dataset+'.csv', encoding="utf-8", index=False)

    df.to_csv('data/CleanData/'+dataset+'_'+str(max_points)+'_0'+'.csv', encoding="utf-8", index=False)


    df.to_csv('../individually-fair-k-clustering-main/data/'+dataset+'_'+str(max_points)+'_0'+'.csv', encoding="utf-8", index=False)
