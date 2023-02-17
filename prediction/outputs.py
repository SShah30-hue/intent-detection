import os
import logging
import json
import configparser as cp
from tqdm import tqdm
import numpy as np
import torch
import uuid as id
import pandas as pd
from modelling import JointBERT 
from utils import load_tokenizer, get_intent_labels, start_connection
from data_loader import load_examples 

logger = logging.getLogger(name="Intent Detection")

config = cp.ConfigParser(interpolation=None)
config.read("C:/IntentDetection/intent-detection-fournet-v2/config.ini")

model_dir = config.get("Misc", "model_dir")
result_dir = config.get("Misc", "result_dir")


def get_args():
    return torch.load(os.path.join(model_dir, 'training_args.bin'))


def load_model(device):
    # Check whether model exists
    if not os.path.exists(model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = JointBERT.from_pretrained(model_dir, intent_label_lst=get_intent_labels())
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")
    
    return model

def predict():
    # load model and args
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    logger.info(args)

    intent_label_lst = get_intent_labels()
    tokenizer = load_tokenizer()
    data_loader, original_words, transcription_id, \
    file_name, chunk_num, channel, duration = load_examples(tokenizer, mode="predict")


    top1_preds = []
    top1_label = []
    top3_probs = []
    top3_preds = []
    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "intent_label_ids": None} 
            inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            _, intent_logits = outputs[:2]
            top_prob, top_class = intent_logits.topk(3, dim=1)
        
            # Top 1 Prediction
            top1_class = torch.argmax(intent_logits, axis=1).squeeze().detach().cpu().tolist()
            for top1_cls in top1_class:
                top1_preds.append(intent_label_lst[top1_cls])
                top1_label.append(top1_cls)

            # Top 3 Predictions
            top3_class = top_class.detach().squeeze().detach().cpu().tolist()
            top3_prob = top_prob.squeeze().detach().cpu().tolist()
            for idx, idx2 in zip(top3_class, top3_prob):
                labels = [intent_label_lst[i] for i in idx]
                probs = [np.round(p, 2) for p in idx2]
                top3_preds.append(', '.join(labels))
                top3_probs.append(str(probs))
                

    # Adding uuid to all file names
    uuid_dict = {}
    for file in file_name:
        uuid_dict.update({file: id.uuid4()})
    uuid_list = []
    for file in file_name:
        uuid_list.append(uuid_dict.get(file))
    
    results = {}
    results['uuid'] = uuid_list
    results['transcription_id'] = transcription_id
    results['transcript'] = original_words
    results['file_name'] = file_name       
    results['chunk_num'] = chunk_num
    results['channel'] = channel
    results['duration'] = duration
    results['predicted_label'] = top1_label
    results['predicted_intent'] = top1_preds
    results['top3_intents'] = top3_preds
    results['top3_probability'] = top3_probs

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(result_dir, 'predictions.csv'), index=False)

    # Connect to db and insert data from df
    engine, connection, metadata = start_connection()
    df.to_sql('intent', con=engine, if_exists='append', index=False)



