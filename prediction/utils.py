from transformers import BertTokenizer
from azure.storage.blob import BlobClient, BlobServiceClient, __version__
import os
import random
import logging
import datetime
import torch
import numpy as np
import pandas as pd
import sqlalchemy as db
import configparser as cp


config = cp.ConfigParser(interpolation=None)
config.read("C:/IntentDetection/intent-detection-fournet-v2/config.ini")

seed = config.getint("Model", "seed")
data_dir = config.get("Misc", "data_dir")
model_dir = config.get("Misc", "model_dir")
known_cls_ratio = config.getfloat("Model", "known_cls_ratio")
chunk_num = config.getint("Misc", "chunk_num")
file = config.get("Misc", "file")


def get_intent_labels():
    return [label.strip() for label in open(os.path.join(model_dir, 'intent_label.txt'), 'r', encoding='utf-8')]


def load_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')


def init_logger():
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    file_name = f"training_{time}.log"

    logging.basicConfig(filename=(os.path.join(model_dir, file_name)),
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    
    logger = logging.getLogger("Intent Detection")
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger

def start_connection():

    db_user = config.get("SQL", "user")
    db_pwd = config.get("SQL", "password")
    db_host = config.get("SQL", "host")
    db_port = config.get("SQL", "port")
    db_name = config.get("SQL", "database")


    # Setting up mySQL connection string and connection to database
    connection_str = f'mysql+pymysql://{db_user}:{db_pwd}@{db_host}:{db_port}/{db_name}'
    ssl_args = {"ssl_ca": "ssl_ca.pem"}
    engine = db.create_engine(connection_str, connect_args=ssl_args)
    connection = engine.connect()
    metadata = db.MetaData(bind=engine)

    return engine, connection, metadata

def data_download():
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir) 

    engine, connection, metadata = start_connection()

    # Reading the data that contain 1 and 2 in the chunk_num column
    df = pd.read_sql("SELECT * FROM transcription where chunk_num < {}".format(chunk_num), connection)

    if(df.empty == False):
        df = df.sort_values('transcription_id')

    # saving as tsv file
    df.to_csv("{}/{}.tsv".format(data_dir, file), sep="\t", index=False)
    file_path = os.path.join(data_dir, file)
    print("Downloaded blob to: " + file_path)

def model_download(connect_str, container_name):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir) 
    
    # Create the BlobServiceClient object which will be used to create a container client
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # Load the container where train and dev data is stored
    container_client = blob_service_client.get_container_client(container_name)

    # List the blobs in the container
    blob_list = container_client.list_blobs()

    for blob in blob_list:
        # Download the blob to a local file
        download_file_path = os.path.join(model_dir, blob.name)
        with open(download_file_path, "wb") as download_file:
            download_file.write(container_client.download_blob(blob.name).readall())
            print("Downloaded blob to: " + download_file_path)