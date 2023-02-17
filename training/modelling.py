from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
import configparser as cp

config = cp.ConfigParser(interpolation=None)
config.read("C:/IntentDetection/intent-detection-fournet-v2/config.ini")

dropout_rate = config.getfloat("Model", "dropout_rate")


class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class JointBERT(BertPreTrainedModel):
    def __init__(self, config, intent_label_lst):
        super(JointBERT, self).__init__(config)
        self.num_intent_labels = len(intent_label_lst)
        self.bert = BertModel(config=config) 

        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, dropout_rate)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids) 
        sequence_output = outputs[0]
        pooled_output = outputs[1] 

        intent_logits = self.intent_classifier(pooled_output)
        total_loss = 0

        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss

        outputs = (intent_logits,) + outputs[2:] 
        outputs = (total_loss,) + outputs

        return outputs 
    



