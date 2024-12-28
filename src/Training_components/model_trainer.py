import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src.loggers import logging
from src.exceptions import CustomeException
from src.utils import save_obj
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainerCongif:
    trainer_model_file_path=os.path.join("artifact","model.pkl")



class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerCongif()


    def initiate_model_trainer(self,train_data,test_data):
     

        logging.info("entered the model intiater")

        X_train,Y_train,X_test,Y_test=(
              
              train_data[:,:-1],
              train_data[:,-1],
              test_data[:,:-1],
              test_data[:,-1]

            )

        logging.info("entered the model prediction")
        model= LinearRegression()
        model.fit(X_train,Y_train,sample_weight=None)
        y_train_predict=model.predict(X_train)
        y_test_predict=model.predict(X_test)

        train_predict_score=r2_score(Y_train,y_train_predict)
        test_predict_score=r2_score(Y_test,y_test_predict)
            
        #linear_model=[test_predict_score]
            
        save_obj (

                file_path=self.model_trainer_config.trainer_model_file_path,
                obj=model

            )

        predicted=model.predict(X_test)
        r2_Score=r2_score(Y_test,predicted)
        return r2_Score
    
    
            