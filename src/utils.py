import os
import sys
import dill
import pickle
from src.exceptions import CustomeException
from src.loggers import logging
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV



def save_obj(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        logging.info("directory path created")

        with open(file_path,"wb") as file_obj:
          pickle.dump(obj,file_obj)
    
    except Exception as e:
       raise CustomeException(e,sys)
    


def load_file(file_path):
        try:
          with open(file_path,"rb") as file_obj:
             return pickle.load(file_obj)
        except Exception as e:
            raise CustomeException(e,sys)