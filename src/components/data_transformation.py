from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import os,sys
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    
    
# Data Ingestionconfig class
class DataTransformation:
    def __init__(self):
        self.data_tranformation_config=DataTransformationConfig()
        
    def get_data_transformation_obj(self):
        try:
            logging.info('Data initiated transformation')
            categorical_cols=['cut','color','clarity']
            numerical_cols=['carat','depth','table','x','y','z']
            
            # Define the custom ranking for each ordinal variable 
            cut_categories=["Fair","Good","Very Good","Premium","Ideal"]
            colour_categories=["D" ,"E" ,"F" , "G" ,"H", "I", "J"]
            clarity_categories=["I1","SI2" ,"SI1" ,"VS2" , "VS1" , "VVS2" , "VVS1" ,"IF"]
            
            num_pipeline=Pipeline(
            steps=[
             ("imputer",SimpleImputer(strategy='median')),
            ('scaler',StandardScaler())]
            )
            cate_pipeline=Pipeline(
            steps=[
            ("imputer",SimpleImputer(strategy='most_frequent')),
            ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,colour_categories,clarity_categories])),
            ('scaler',StandardScaler())]
            )

            preprocessor=ColumnTransformer([
            
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cate_pipeline,categorical_cols)
            ])
            return preprocessor
            logging.info('Pipeline completed')
            
        except Exception as e:
            logging.info('error in data transformation')
            raise CustomException(e,sys)
            
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info('read train and test data completed')
            logging.info(f'train dataframe head :\n{train_df.head().to_string()}')
            logging.info(f'test dataframe head :\n{test_df.head().to_string()}')
            
            logging.info('obtaining preprocessing object')
            
            preprocessing_obj=self.get_data_transformation_obj()
            
            target_column_name='price'
            drop_columns=[target_column_name,'id']
            
            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            #apply the transformation
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            logging.info('Applying preprocessing object on training and testing datasets.')
            
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            save_object(file_path=self.data_tranformation_config.preprocessor_obj_file_path,
                        obj=preprocessing_obj)
            
            logging.info('preprocessor pickle is created and saved')
            return(train_arr,test_arr,self.data_tranformation_config.preprocessor_obj_file_path)
        
        except Exception as e:
            logging.info('error in data transformation')
            raise CustomException(e,sys)