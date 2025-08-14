import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" # datetime now, as a log file
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE) # log path: join current working directory, logs folder with log file
os.makedirs(logs_path,exist_ok=True) # make directory logs_path

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)  

logging.basicConfig( # basic config
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s", # timestamp, linenum, name, message
    level=logging.INFO, 


)

# for testing
# if __name__=="__main__":
#     logging.info("logging has started")