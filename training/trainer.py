from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import compute_metrics, get_intent_labels
from transformers import BertConfig
from modelling import JointBERT
import configparser as cp
import os
import logging
import numpy as np
import torch


logger = logging.getLogger(name="Intent Detection")


config = cp.ConfigParser(interpolation=None)
config.read("C:/IntentDetection/intent-detection-fournet-v2/config.ini")

model_dir = config.get("Misc", "model_dir")
ignore_index = config.get("Model", "ignore_index")
max_steps = config.getint("Model", "max_steps")
train_batch_size = config.getint("Model", "train_batch_size")
eval_batch_size = config.getint("Model", "eval_batch_size")
gradient_accumulation_steps = config.getint("Model", "gradient_accumulation_steps")
num_train_epochs = config.getfloat("Model", "num_train_epochs")
weight_decay = config.getfloat("Model", "weight_decay")
learning_rate = config.getfloat("Model", "learning_rate")
adam_epsilon = config.getfloat("Model", "adam_epsilon")
max_grad_norm = config.getfloat("Model", "max_grad_norm")
warmup_steps = config.getint("Model", "warmup_steps")
logging_steps = config.getint("Model", "logging_steps")
save_steps = config.getint("Model", "save_steps")


class Trainer(object):
    def __init__(self, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.intent_label_lst = get_intent_labels()
        self.pad_token_label_id = ignore_index

        self.config = BertConfig.from_pretrained('bert-base-uncased')
        self.model = JointBERT.from_pretrained('bert-base-uncased', config=self.config, intent_label_lst=self.intent_label_lst) 

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.num_train_epochs = num_train_epochs

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        if max_steps > 0:
            t_total = max_steps
            self.num_train_epochs = max_steps // (len(train_dataloader) // gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // gradient_accumulation_steps * self.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num intents = %d", len(self.intent_label_lst))
        logger.info("  Known intents = %s", self.intent_label_lst)
        logger.info("  Num Epochs = %d", self.num_train_epochs)
        logger.info("  Total train batch size = %d", train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", logging_steps)
        logger.info("  Save steps = %d", save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch) 

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3]}
                 
                inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                loss = outputs[0]

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    optimizer.step()
                    scheduler.step() 
                    self.model.zero_grad()
                    global_step += 1

                    if logging_steps > 0 and global_step % logging_steps == 0:
                        self.evaluate("dev")

                    if save_steps > 0 and global_step % save_steps == 0:
                        self.save_model()

                if 0 < max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        intent_preds = None
        out_intent_label_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3]} 
                
                inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                tmp_eval_loss, intent_logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
                out_intent_label_ids = inputs['intent_label_ids'].detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                out_intent_label_ids = np.append(
                    out_intent_label_ids, inputs['intent_label_ids'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        intent_preds = np.argmax(intent_preds, axis=1)
        total_result = compute_metrics(intent_preds, out_intent_label_ids)
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(model_dir)

        # Save training arguments together with the trained model
        args = dict(config.items('Model'))
        torch.save(args, os.path.join(model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", model_dir)

        # Save intent_label_lst
        with open(os.path.join(model_dir, 'intent_label.txt'), 'w', encoding='utf-8') as f_w:
            for intent in self.intent_label_lst:
                f_w.write(intent + '\n')

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = JointBERT.from_pretrained(model_dir, intent_label_lst=self.intent_label_lst)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
