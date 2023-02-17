from trainer import Trainer
from utils import data_download, save_model_to_azure, init_logger, load_tokenizer, set_seed
from data_loader import load_and_cache_examples
import logging
import configparser as cp


config = cp.ConfigParser(interpolation=None)
config.read("C:/IntentDetection/intent-detection-fournet-v2/config.ini")

connect_str = config.get("Azure", "Connection_String")
data_container_name = config.get("Azure", "data_container_name")
model_container_name = config.get("Azure", "model_container_name")
model_dir = config.get("Misc", "model_dir")


logger = init_logger()
logger = logging.getLogger(name="Intent Detection")

logger.info('Downloading Data...')
data_download(connect_str, data_container_name)
logger.info('Download Complete...')

set_seed()
tokenizer = load_tokenizer()
train_dataset = load_and_cache_examples(tokenizer, mode="train")
dev_dataset = load_and_cache_examples(tokenizer, mode="dev")
test_dataset = load_and_cache_examples(tokenizer, mode="test")

trainer = Trainer(train_dataset, dev_dataset, test_dataset)
trainer.train()
logger.info('Training Finished...')
trainer.load_model()
trainer.evaluate("test")

logger.info('Saving best model to azure...')
save_model_to_azure(connect_str, model_container_name, model_dir) 
logger.info('Best model saved to "intent-analysis-model" azure blob storage')



