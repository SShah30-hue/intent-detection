import configparser as cp
from outputs import predict
from utils import model_download, data_download, init_logger 

config = cp.ConfigParser(interpolation=None)
config.read("C:/IntentDetection/intent-detection-fournet-v2/config.ini")

connect_str = config.get("Azure", "Connection_String")
model_container_name = config.get("Azure", "model_container_name")

logger = init_logger()

logger.info('Intent Detection')
logger.info('Downloading Data...')
data_download()
logger.info('Downloading Model...')
#model_download(connect_str, model_container_name)
logger.info('Download Complete...')
predict()
logger.info('Prediction Saved')

