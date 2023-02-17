from transformers import BertTokenizer
from azure.storage.blob import BlobClient, BlobServiceClient, __version__
import os
import random
import logging
import datetime
import torch
import numpy as np
import pandas as pd
import configparser as cp

config = cp.ConfigParser(interpolation=None)
config.read("C:/IntentDetection/intent-detection-fournet-v2/config.ini")

seed = config.getint("Model", "seed")
data_dir = config.get("Misc", "data_dir")
model_dir = config.get("Misc", "model_dir")
known_cls_ratio = config.getfloat("Model", "known_cls_ratio")


def get_intent_labels():
    train_data_dir = os.path.join(data_dir, "train.tsv")
    df = pd.read_csv(train_data_dir, sep='\t')
    all_label_list = df.label.unique()
    n_known_cls = round(len(all_label_list) * known_cls_ratio)
    known_label_list = np.random.choice(np.array(all_label_list), n_known_cls, replace=False)
    known_label_list = list(known_label_list)
    known_label_list.append("UNK")
    intent_vocab = sorted(list(known_label_list))

    return [label.strip() for label in intent_vocab]


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


def set_seed():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(intent_preds, intent_labels): 
    assert len(intent_preds) == len(intent_labels)
    results = {}
    intent_result = get_intent_acc(intent_preds, intent_labels)
    results.update(intent_result)

    return results

def get_intent_acc(preds, labels):
    acc = (preds == labels).mean()

    return {
        "intent_acc": acc
    }


def get_sentence_frame_acc(intent_preds, intent_labels):
    """For the cases that intent and all the slots are correct (in one sentence)"""
    # Get the intent comparison result
    intent_result = (intent_preds == intent_labels)
    sementic_acc = intent_result.mean()

    return {
        "sementic_frame_acc": sementic_acc
    }

def data_download(connect_str, container_name):
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir) 

    # Create the BlobServiceClient object which will be used to create a container client
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # Load the container where train and dev data is stored
    container_client = blob_service_client.get_container_client(container_name)

    # List the blobs in the container
    blob_list = container_client.list_blobs()

    for blob in blob_list:
        # Download the blob to a local file
        download_file_path = os.path.join(data_dir, blob.name)
        with open(download_file_path, "wb") as download_file:
            download_file.write(container_client.download_blob(blob.name).readall())
            print("Downloaded blob to: " + download_file_path)

def save_model_to_azure(connect_str, model_container_name, model_dir):
    
    # List the files in the model dir
    for file in os.listdir(model_dir):
        try:
            # Upload the blob to a blob container
            upload_file_path = os.path.join(model_dir, file)
            # Create the BlobClient object which will be used to create a container client
            blob_client = BlobClient.from_connection_string(connect_str, model_container_name, file, max_block_size=4*1024*1024, max_single_put_size=16*1024*1024)
            with open(upload_file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True, max_concurrency=1, timeout=1800)
                print("Uploading to azure storage as blob: " + file)

        except Exception as ex:
            print('Error while uploading files to azure blob storage')
            print(ex) 

