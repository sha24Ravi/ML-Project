import os
import sys
import dill
from src.exceptions import CustomeException
from src.loggers import logging
import pandas as pd
import numpy as np



def save_obj(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        logging.info("directory path created")

        with open(file_path,"wb") as file_obj:
          dill.dump(obj,file_obj)
    
    except Exception as e:
       raise CustomeException(e,sys)