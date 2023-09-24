import pandas as pd
import numpy as np
import datetime
import math
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
# Support Vector Regression (SVR)
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report

models = []
train_mse = []
test_mse = []
train_r2 = []
test_r2 = []
train_rmse = []
test_rmse = []

train_accuracy = []  # Classification metric
test_accuracy = []   # Classification metric
train_confusion_matrix = []  # Classification metric
test_confusion_matrix = []   # Classification metric
def prediction_model(data:pd.DataFrame,
                     model: object,
                     model_name:str,
                     train_end:datetime,
                     test_start:datetime,
                     problem_type: str = 'regression'
                     ) -> (pd.DataFrame, pd.DataFrame):
    
    """
    Train and evaluate a predictive model.

    Parameters:
    - data (pd.DataFrame): The dataset containing features and labels.
    - model (object): The machine learning model to train and evaluate.
    - model_name (str): The name of the model for documentation.
    - train_end (datetime): The end date for the training data.
    - test_start (datetime): The start date for the test data.
    - problem_type (str): 'regression' or 'classification'.

    Returns:
    - train_df (pd.DataFrame): DataFrame with training predictions.
    - test_df (pd.DataFrame): DataFrame with test predictions.
    """

    # X = data[['close', 'SMA', 'EMA', 'ROC', 'RSI']]

    X = data.drop(columns=['label'])
    y = data['label']

    X_train = X[:train_end]
    X_test = X[test_start:]
    y_train = y[:train_end]
    y_test = y[test_start:]

    model.fit(X_train,y_train)

    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)
    
    print(f"========MODEL : {model_name}==================")
    models.append(model_name)
    # metrics
    # Performance Evaluation
    # print("=============During Training===================")
    # print(f'Mean Squared Error (MSE): {mean_squared_error(y_train, y_pred_train)}',
    #       f'R2 : {r2_score(y_train, y_pred_train)} RMSE: {math.sqrt(mean_squared_error(y_train, y_pred_train))}')
    
    # print("=============During Test=======================")
    # print(f'Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred)}',
    #       f'R2 : {r2_score(y_test, y_pred)} RMSE: {math.sqrt(mean_squared_error(y_test, y_pred))}')
    if problem_type == 'regression':

        train_mse.append(mean_squared_error(y_train, y_pred_train))
        test_mse.append(mean_squared_error(y_test, y_pred))

        train_r2.append(r2_score(y_train, y_pred_train))
        test_r2.append(r2_score(y_test, y_pred))

        train_rmse.append(math.sqrt(mean_squared_error(y_train, y_pred_train)))
        test_rmse.append(math.sqrt(mean_squared_error(y_test, y_pred)))
    elif problem_type == 'classification':
        # Classification Metrics
        train_accuracy.append(accuracy_score(y_train, y_pred_train))
        test_accuracy.append(accuracy_score(y_test, y_pred))

        train_confusion_matrix.append(confusion_matrix(y_train, y_pred_train))
        test_confusion_matrix.append(confusion_matrix(y_test, y_pred))
    
    X_train['prediction'] = y_pred_train
    X_test['prediction'] = y_pred
    
    X_train.reset_index(drop = False, inplace = True)
    X_test.reset_index(drop = False, inplace = True)

    # X_train.drop(['close' ,'SMA' ,'EMA' , 'ROC' , 'RSI' ], axis = 1, inplace = True)
    # X_test.drop(['close' ,'SMA' ,'EMA' , 'ROC' , 'RSI' ], axis = 1, inplace = True)
    
    

    #train_df
    X_train.set_index('datetime', inplace=True)
    y_train = pd.DataFrame(y_train)
    train_df = pd.merge(y_train, X_train, left_index=True, right_index=True)

    #test_df
    X_test.set_index('datetime', inplace=True)
    y_test = pd.DataFrame(y_test)
    test_df = pd.merge(y_test, X_test, left_index=True, right_index=True)
    
    return train_df, test_df
