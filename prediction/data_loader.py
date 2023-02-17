from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import torch
import os
import csv
import logging
import copy
import json
import configparser as cp

logger = logging.getLogger(name="Intent Detection")

config = cp.ConfigParser(interpolation=None)
config.read("C:/IntentDetection/intent-detection-fournet-v2/config.ini")

max_seq_length = config.getint("Model", "max_seq_length")
data_dir = config.get("Misc", "data_dir")
file = config.get("Misc", "file")
ignore_index = config.get("Model", "ignore_index")
predict_batch_size = config.getint("Model", "predict_batch_size")



class InputExample(object):
    """A single training/test example for simple sequence classification.

        Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        transcription_id, file_name, chunk_num, channel, duration (Optional): additional columns. 
    """

    def __init__(self, guid, words, original_words=None, transcription_id=None, file_name=None, 
                 chunk_num=None, channel=None, duration=None):  
        self.guid = guid
        self.words = words
        self.original_words = original_words
        self.transcription_id = transcription_id
        self.file_name = file_name
        self.chunk_num = chunk_num
        self.channel = channel
        self.duration = duration

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids

class JointProcessor(object):
    
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            next(reader)  # skip the first line - column names
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            words = line[2].split()
            original_words = line[2]
            transcription_id = line[1]
            file_name = line[3]
            chunk_num = line[4]
            channel = line[5]
            duration = line[6].split(' ')[2]

            examples.append(InputExample(guid=guid, words=words, original_words=original_words, transcription_id=transcription_id, 
                                         file_name=file_name, chunk_num=chunk_num, channel=channel, duration=duration))
        return examples
        
    def get_examples(self, data_dir, mode):
        if mode == 'predict':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "{}.tsv".format(file))), "predict")
        else:
            raise Exception("For FourNet service desk calls only")
        

def convert_examples_to_features(examples, max_seq_length, tokenizer,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    original_words = []
    transcription_id = []
    file_name = []
    chunk_num = []
    channel = []
    duration = []
    
    for example in examples:
        # Tokenize word by word (for NER)
        tokens = []
        for word in example.words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        original_words.append(example.original_words)
        transcription_id.append(example.transcription_id)
        file_name.append(example.file_name)
        chunk_num.append(example.chunk_num)
        channel.append(example.channel)
        duration.append(example.duration)
        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids))

    return features, original_words, transcription_id, file_name, chunk_num, channel, duration

def load_examples(tokenizer, mode):
    processor = JointProcessor()
    logger.info("Creating features from dataset file at %s", data_dir)
    if mode == "predict":
        examples = processor.get_examples(data_dir, "predict")
    else:
        raise Exception("For FourNet service desk calls and test mode only")
    
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = ignore_index
    features, original_words, transcription_id, \
    file_name, chunk_num, channel, duration = convert_examples_to_features(examples, max_seq_length, tokenizer, pad_token_label_id=pad_token_label_id)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=predict_batch_size)

    return data_loader, original_words, transcription_id, file_name, chunk_num, channel, duration