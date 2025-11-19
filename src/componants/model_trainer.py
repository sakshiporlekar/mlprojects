import os
import sys
from dataclasses import dataclass

from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj,evalute_model

@dataclass
class ModelTrainerconfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerconfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('spliting training and test input data')
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                    "linear":LinearRegression(),
                    "lasso":Lasso(),
                    "ridge":Ridge(),
                    "kneigbor":KNeighborsRegressor(),
                    "decisiontree":DecisionTreeRegressor(),
                    "randomforest":RandomForestRegressor(),
                    "adaboost":AdaBoostRegressor(),
                    "xgboost":XGBRegressor()
                    }
            model_report=dict=evalute_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            
            #to get best model score from dict
            best_model_score=max(sorted(model_report.values()))

            #to get best model name from dict
            base_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[base_model_name]

            if best_model_score<0.6:
                raise CustomException('No best model found')
            logging.info('best found model on both training and testing dataset')

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(x_test)
            r2_score_val=r2_score(y_test,predicted)

            return r2_score_val


        except Exception as e:
            raise CustomException(e,sys)

    
