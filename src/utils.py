import os
import sys

import numpy as np 
import pandas as pd
import dill
# import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models): #,param
    try:
        report = {}

        model_list = []
        r2_list =[]
        
        for i in range(len(list(models))):
            model = list(models.values())[i] # get each and every model 
           # para=param[list(models.keys())[i]]

            # gs = GridSearchCV(model,para,cv=3)
            # gs.fit(X_train,y_train)

            # model.set_params(**gs.best_params_)
            model.fit(X_train,y_train) # train model

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

                # Evaluate Train and Test dataset
            model_train_mae , model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)

            model_test_mae , model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score # append to report

            #### Also print it out
            print(list(models.keys())[i])
            model_list.append(list(models.keys())[i])
    
            print('Model performance for Training set')
            print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
            print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
            print("- R2 Score: {:.4f}".format(model_train_r2))

            print('----------------------------------')
    
            print('Model performance for Test set')
            print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
            print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
            print("- R2 Score: {:.4f}".format(model_test_r2))
            r2_list.append(model_test_r2)
    
            print('='*35)
            print('\n')

        return report

    except Exception as e:
        raise CustomException(e, sys)

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr
    
# def load_object(file_path):
#     try:
#         with open(file_path, "rb") as file_obj:
#             return pickle.load(file_obj)

#     except Exception as e:
#         raise CustomException(e, sys)

def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square