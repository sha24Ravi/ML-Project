import os
import sys
import pandas as pd
from src.exceptions import CustomeException
from src.loggers import logging
from src.Training_components import data_transformation,model_trainer
from  sklearn.model_selection import train_test_split
from dataclasses import dataclass



@dataclass
class dataIngestion_config:
    train_data_path=os.path.join("artifact","train.csv")
    test_data_path=os.path.join("artifact","test.csv")
    raw_data_path=os.path.join("artifact","raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=dataIngestion_config()


    def initiate_dataIngestion(self):
        logging.info("entered the data ingestion intiater")

        try:
            df= pd.read_csv('Notebook/Data/stud.csv')
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("started train test split")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)


            return(

                self.ingestion_config.test_data_path,
                self.ingestion_config.train_data_path
            )
        except  Exception as e:
            raise CustomeException(e,sys)        

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_dataIngestion()
    dataTransformation=data_transformation.DataTransformation()
    train_arr,test_arr,_=dataTransformation.intiate_data_tranform(train_data,test_data)
    modelTrainer=model_trainer.ModelTrainer()
    score= modelTrainer.initiate_model_trainer(train_arr,test_arr)
    print(score)

