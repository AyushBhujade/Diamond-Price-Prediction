import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_ingestion import datainjestion

from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainer
if __name__=='__main__':
    obj=datainjestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    print(train_data_path,test_data_path)
    
    data_transformation=DataTransformation()
    
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    
    model_train=ModelTrainer()
    model_train.initiate_model_training(train_arr,test_arr)