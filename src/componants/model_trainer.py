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
            
            params={
                'linear':{},
                 'lasso':{'alpha':[0.0001, 0.001, 0.01, 0.1, 1, 10],'max_iter': [1000, 2000],'selection': ['cyclic', 'random'] },
                 'ridge':{'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],'max_iter': [1000, 2000,3000]},
                 'kneigbor':{'n_neighbors': [1, 3, 5, 7, 9, 11, 15],'algorithm': ['auto', 'ball_tree', 'kd_tree'],'p':[1,2]},
                 'decisiontree':{'max_depth': [3, 5, 7, 10, 15, 20, None],'min_samples_split': [2, 5, 10, 15]},
                 'randomforest':{'n_estimators': [50, 100, 200, 300, 500],'max_depth': [5, 10, 15, 20],'min_samples_split': [2, 5, 10]},
                 'adaboost':{'n_estimators': [50, 100, 200, 300],'learning_rate': [0.001, 0.01, 0.1, 0.5, 1]},
                 'xgboost':{'n_estimators': [200, 300, 500],'learning_rate': [0.05, 0.1, 0.2]}
                 
            }
            model_report=evalute_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,param=params)
            
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

            return r2_score_val,best_model


        except Exception as e:
            raise CustomException(e,sys)

    
