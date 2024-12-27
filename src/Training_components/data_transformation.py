import os
import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exceptions import CustomeException
from src.loggers import logging
from src.utils import save_obj
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationConfig:
   preprocessor_path=os.path.join("artifact","preprocessor.pkl")



class DataTransformation:
   def __init__(self):
      self.data_transformation_path=DataTransformationConfig()


   def get_data_transformation_object(self):
         
      try:

            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            numerical_pipeline=Pipeline(
               
               steps=[
                  
                  ("impute",SimpleImputer(strategy="median")),
                  ("scaler",StandardScaler(with_mean=False))
               ]
            )


            categorical_pipeline=Pipeline(
               
               steps=[
                  ("impute",SimpleImputer(strategy="most_frequent")),
                  ("one_hot_encoder",OneHotEncoder()),
                  ("scaler",StandardScaler(with_mean=False))
               ]
            )
      
            preprocessor=ColumnTransformer(
              [ 
               ("numerical_pipeline",numerical_pipeline,numerical_columns),
               ("categorical_pipeline",categorical_pipeline,categorical_columns)

              ] 
            )



            return preprocessor
            
      except Exception as e:
            raise CustomeException(e,sys)



   def intiate_data_tranform(self,train_data,test_data):

      try: 

         train_df=pd.read_csv(train_data)  
         test_df=pd.read_csv(test_data)    
  
         logging.info("read train and test data")

         preprocessor_obj=self.get_data_transformation_object()

         logging.info("read preprocessor object")

         #the label used for prediction using train data
         target_coloumn="math_score"
         #this is the feature used for trianing and prediction using train data
         numerical_columns = ["writing_score", "reading_score"]
         input_feature_train_df=train_df.drop(columns=[target_coloumn],axis=1)
         target_train_df=train_df[target_coloumn]

         #simillarly for the test data
         input_feature_test_df=test_df.drop(columns=[target_coloumn],axis=1)
         target_test_df=test_df[target_coloumn]
         
         logging.info("entering preprocessing stage")

         input_feature_train_df_arr=preprocessor_obj.fit_transform(input_feature_train_df)
         input_feature_test_df_arr=preprocessor_obj.transform(input_feature_test_df)

         #now combine both the transformed input_features and target/label data of training set

         train_arr=np.c_[input_feature_train_df_arr,np.array(target_train_df)]

         test_arr=np.c_[input_feature_test_df_arr,np.array(target_test_df)]


         save_obj(

            file_path=self.data_transformation_path.preprocessor_path,
            obj=preprocessor_obj
         )

         return (
          
          train_arr,
          test_arr,
          self.data_transformation_path.preprocessor_path,

         )

      except Exception as e:
          raise CustomeException(e,sys)






    